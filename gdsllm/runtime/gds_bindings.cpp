/*
 * GdsLLM — Python bindings for GDS I/O module
 *
 * Exposes cuFile operations to Python via pybind11, with torch::Tensor
 * integration using torch::from_blob with custom deleters.
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>

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
}
