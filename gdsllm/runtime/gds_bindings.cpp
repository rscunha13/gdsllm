/*
 * GdsLLM — Python bindings for GDS I/O module
 *
 * Exposes cuFile operations to Python via pybind11, with torch::Tensor
 * integration using torch::from_blob with custom deleters.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "gguf_dequant.h"
#include "fused_gemv.h"

namespace py = pybind11;

// ─── Forward declarations from gds_io.cu ──────────────────────────────────

struct GdsBuffer;

void gds_init();
void gds_shutdown();
bool gds_is_driver_open();
GdsBuffer* gds_alloc(size_t size, int device_id);
void gds_free(GdsBuffer* buf);
ssize_t gds_read(GdsBuffer* buf, const char* filepath,
                 size_t size, off_t file_offset, off_t buf_offset);

// Access GdsBuffer fields (defined in gds_io.cu)
extern "C" {
    // We need access to GdsBuffer internals for the bindings.
    // Since the struct is defined in gds_io.cu and included here via
    // forward declaration, we redefine it identically.
}

struct GdsBuffer {
    void* dev_ptr;
    size_t size;
    int device_id;
    bool registered;
};

// ─── Buffer wrapper with shared_ptr for Python reference counting ─────────

class GdsBufferHandle {
public:
    GdsBufferHandle(size_t size, int device_id) {
        buf_ = gds_alloc(size, device_id);
    }

    ~GdsBufferHandle() {
        if (buf_) {
            gds_free(buf_);
            buf_ = nullptr;
        }
    }

    // Non-copyable
    GdsBufferHandle(const GdsBufferHandle&) = delete;
    GdsBufferHandle& operator=(const GdsBufferHandle&) = delete;

    GdsBuffer* raw() { return buf_; }
    void* dev_ptr() { return buf_ ? buf_->dev_ptr : nullptr; }
    size_t size() { return buf_ ? buf_->size : 0; }
    int device_id() { return buf_ ? buf_->device_id : 0; }
    bool registered() { return buf_ ? buf_->registered : false; }

private:
    GdsBuffer* buf_ = nullptr;
};

// ─── Helper: dtype string to torch dtype ──────────────────────────────────

static torch::Dtype parse_dtype(const std::string& dtype_str) {
    if (dtype_str == "float16" || dtype_str == "half") return torch::kFloat16;
    if (dtype_str == "float32" || dtype_str == "float") return torch::kFloat32;
    if (dtype_str == "bfloat16") return torch::kBFloat16;
    if (dtype_str == "int8") return torch::kInt8;
    if (dtype_str == "int32" || dtype_str == "int") return torch::kInt32;
    throw std::runtime_error("Unsupported dtype: " + dtype_str);
}

// ─── Python-facing functions ──────────────────────────────────────────────

/**
 * Load an entire file from NVMe into a new CUDA buffer.
 * Returns a shared GdsBufferHandle that Python can hold.
 */
static std::shared_ptr<GdsBufferHandle> py_load_file(
    const std::string& filepath,
    int64_t size_bytes,
    int device_id = 0
) {
    auto handle = std::make_shared<GdsBufferHandle>(size_bytes, device_id);
    ssize_t bytes_read = gds_read(
        handle->raw(), filepath.c_str(), size_bytes, 0, 0
    );
    if (bytes_read < static_cast<ssize_t>(size_bytes)) {
        throw std::runtime_error(
            "py_load_file: expected " + std::to_string(size_bytes) +
            " bytes, got " + std::to_string(bytes_read));
    }
    return handle;
}

/**
 * Create a torch::Tensor view into a GdsBufferHandle at a given offset.
 * The tensor does NOT own the memory — the GdsBufferHandle does.
 * Python must keep a reference to the handle while using the tensor.
 */
static torch::Tensor py_view_tensor(
    std::shared_ptr<GdsBufferHandle> handle,
    std::vector<int64_t> shape,
    const std::string& dtype_str,
    int64_t offset
) {
    if (!handle || !handle->dev_ptr()) {
        throw std::runtime_error("py_view_tensor: invalid buffer handle");
    }

    torch::Dtype dtype = parse_dtype(dtype_str);
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::kCUDA, handle->device_id());

    // Compute pointer at offset
    void* ptr = static_cast<char*>(handle->dev_ptr()) + offset;

    // Create tensor that shares memory with the buffer.
    // The custom deleter captures a copy of the shared_ptr, keeping
    // the buffer alive as long as any tensor references it.
    auto ref = handle;  // capture shared_ptr by value in deleter
    torch::Tensor tensor = torch::from_blob(
        ptr, shape,
        /*deleter=*/[ref](void*) {
            // prevent release by preventing the shared_ptr from going away
        },
        options
    );

    return tensor;
}

/**
 * Convenience: load a file and return a single tensor.
 * Allocates a buffer, reads the file, and wraps it as a tensor.
 * The buffer is freed when the tensor is garbage collected.
 */
static torch::Tensor py_load_tensor(
    const std::string& filepath,
    std::vector<int64_t> shape,
    const std::string& dtype_str,
    int64_t file_offset,
    int64_t size_bytes,
    int device_id = 0
) {
    auto handle = std::make_shared<GdsBufferHandle>(size_bytes, device_id);
    ssize_t bytes_read = gds_read(
        handle->raw(), filepath.c_str(), size_bytes, file_offset, 0
    );
    if (bytes_read < static_cast<ssize_t>(size_bytes)) {
        throw std::runtime_error(
            "py_load_tensor: expected " + std::to_string(size_bytes) +
            " bytes, got " + std::to_string(bytes_read));
    }

    torch::Dtype dtype = parse_dtype(dtype_str);
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::kCUDA, device_id);

    // Capture handle in deleter to keep buffer alive
    auto ref = handle;
    torch::Tensor tensor = torch::from_blob(
        handle->dev_ptr(), shape,
        [ref](void*) { /* prevent release until tensor is freed */ },
        options
    );

    return tensor;
}

/**
 * Dequantize GGUF block-quantized data from a GDS buffer into an fp16 tensor.
 *
 * Reads raw block data at `offset` within the buffer, runs the appropriate
 * CUDA dequant kernel (Q4_0 or Q8_0), and returns a new fp16 tensor.
 */
static torch::Tensor py_dequant_gguf(
    std::shared_ptr<GdsBufferHandle> handle,
    std::vector<int64_t> shape,
    const std::string& gguf_type,
    int64_t offset,
    c10::optional<torch::Tensor> output_tensor
) {
    if (!handle || !handle->dev_ptr()) {
        throw std::runtime_error("py_dequant_gguf: invalid buffer handle");
    }

    // Compute total number of elements from shape
    int64_t num_elements = 1;
    for (auto s : shape) num_elements *= s;

    if (num_elements % 32 != 0) {
        throw std::runtime_error(
            "py_dequant_gguf: total elements must be a multiple of 32 (block size), got " +
            std::to_string(num_elements));
    }
    size_t num_blocks = num_elements / 32;

    // Source pointer in the GDS buffer
    const void* src = static_cast<const char*>(handle->dev_ptr()) + offset;

    // Use pre-allocated output tensor if provided, otherwise allocate
    torch::Tensor output;
    if (output_tensor.has_value()) {
        output = output_tensor.value();
        // Reshape scratch to match requested shape (must have enough elements)
        if (output.numel() < num_elements) {
            throw std::runtime_error(
                "py_dequant_gguf: scratch buffer too small (" +
                std::to_string(output.numel()) + " < " +
                std::to_string(num_elements) + ")");
        }
        // Return a view with the correct shape (avoids reallocation)
        output = output.flatten().slice(0, 0, num_elements).view(shape);
    } else {
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::kCUDA, handle->device_id());
        output = torch::empty(shape, options);
    }

    __half* dst = reinterpret_cast<__half*>(output.data_ptr());

    // Get current CUDA stream from PyTorch
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(handle->device_id());

    if (gguf_type == "q8_0") {
        launch_dequant_q8_0(src, dst, num_blocks, stream);
    } else if (gguf_type == "q4_0") {
        launch_dequant_q4_0(src, dst, num_blocks, stream);
    } else {
        throw std::runtime_error("py_dequant_gguf: unsupported type '" + gguf_type + "'");
    }

    return output;
}

/**
 * Fused dequant + matrix-vector multiply for decode (batch=1, seq_len=1).
 *
 * Reads Q4_0/Q8_0 blocks directly from the GDS buffer and computes the
 * dot product with the input vector in a single kernel, avoiding the
 * fp16 intermediate allocation entirely.
 *
 * Args:
 *   handle: GDS buffer containing the quantized weight data
 *   x: input tensor [1, in_dim] or [in_dim] fp16
 *   weight_shape: [out_dim, in_dim] shape of the weight matrix
 *   gguf_type: "q4_0" or "q8_0"
 *   offset: byte offset of the weight data within the buffer
 *
 * Returns:
 *   output tensor [1, out_dim] fp16
 */
static torch::Tensor py_fused_dequant_gemv(
    std::shared_ptr<GdsBufferHandle> handle,
    torch::Tensor x,
    std::vector<int64_t> weight_shape,
    const std::string& gguf_type,
    int64_t offset
) {
    if (!handle || !handle->dev_ptr()) {
        throw std::runtime_error("py_fused_dequant_gemv: invalid buffer handle");
    }
    if (weight_shape.size() != 2) {
        throw std::runtime_error("py_fused_dequant_gemv: weight_shape must be [out_dim, in_dim]");
    }

    int out_dim = weight_shape[0];
    int in_dim = weight_shape[1];

    if (in_dim % 32 != 0) {
        throw std::runtime_error(
            "py_fused_dequant_gemv: in_dim must be a multiple of 32, got " +
            std::to_string(in_dim));
    }

    // Flatten input to [in_dim]
    torch::Tensor x_flat = x.contiguous().view({-1});
    if (x_flat.size(0) != in_dim) {
        throw std::runtime_error(
            "py_fused_dequant_gemv: input size " + std::to_string(x_flat.size(0)) +
            " != in_dim " + std::to_string(in_dim));
    }

    // Weight data pointer
    const void* weight_ptr = static_cast<const char*>(handle->dev_ptr()) + offset;

    // Allocate output [1, out_dim]
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(torch::kCUDA, handle->device_id());
    torch::Tensor output = torch::empty({1, out_dim}, options);

    const __half* x_ptr = reinterpret_cast<const __half*>(x_flat.data_ptr());
    __half* y_ptr = reinterpret_cast<__half*>(output.data_ptr());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(handle->device_id());

    if (gguf_type == "q4_0") {
        launch_fused_q4_0_gemv(weight_ptr, x_ptr, y_ptr, out_dim, in_dim, stream);
    } else if (gguf_type == "q8_0") {
        launch_fused_q8_0_gemv(weight_ptr, x_ptr, y_ptr, out_dim, in_dim, stream);
    } else {
        throw std::runtime_error("py_fused_dequant_gemv: unsupported type '" + gguf_type + "'");
    }

    return output;
}

// ─── Module definition ────────────────────────────────────────────────────

PYBIND11_MODULE(gds_io_ext, m) {
    m.doc() = "GdsLLM GDS I/O extension — cuFile-based NVMe-to-VRAM loading";

    // GDS driver lifecycle
    m.def("init", &gds_init,
          "Initialize the cuFile driver. Call once at startup.");
    m.def("shutdown", &gds_shutdown,
          "Shut down the cuFile driver. Call once at exit.");
    m.def("is_driver_open", &gds_is_driver_open,
          "Returns True if cuFile driver opened successfully.");

    // Buffer handle class
    py::class_<GdsBufferHandle, std::shared_ptr<GdsBufferHandle>>(
        m, "GdsBufferHandle",
        "Handle to a CUDA buffer registered with cuFile."
    )
        .def_property_readonly("size", &GdsBufferHandle::size)
        .def_property_readonly("device_id", &GdsBufferHandle::device_id)
        .def_property_readonly("registered", &GdsBufferHandle::registered);

    // File loading
    m.def("load_file", &py_load_file,
          "Load an entire file from NVMe into a CUDA buffer via GDS.",
          py::arg("filepath"),
          py::arg("size_bytes"),
          py::arg("device_id") = 0);

    // Tensor view from buffer
    m.def("view_tensor", &py_view_tensor,
          "Create a torch.Tensor view into a GdsBufferHandle at an offset.",
          py::arg("handle"),
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("offset"));

    // Convenience: load file directly as tensor
    m.def("load_tensor", &py_load_tensor,
          "Load a region of a file directly into a CUDA tensor via GDS.",
          py::arg("filepath"),
          py::arg("shape"),
          py::arg("dtype") = "float16",
          py::arg("file_offset") = 0,
          py::arg("size_bytes") = 0,
          py::arg("device_id") = 0);

    // GGUF block dequantization
    m.def("dequant_gguf", &py_dequant_gguf,
          "Dequantize GGUF block-quantized data (Q4_0/Q8_0) from a buffer to fp16.",
          py::arg("handle"),
          py::arg("shape"),
          py::arg("gguf_type"),
          py::arg("offset"),
          py::arg("output") = py::none());

    // Fused dequant + GEMV (decode path)
    m.def("fused_dequant_gemv", &py_fused_dequant_gemv,
          "Fused dequant + matrix-vector multiply for Q4_0/Q8_0 (decode only).",
          py::arg("handle"),
          py::arg("x"),
          py::arg("weight_shape"),
          py::arg("gguf_type"),
          py::arg("offset"));
}
