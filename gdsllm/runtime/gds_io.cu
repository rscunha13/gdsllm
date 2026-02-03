/*
 * GdsLLM — GDS I/O Module
 *
 * Low-level CUDA/cuFile operations for loading data directly from
 * NVMe into GPU VRAM using GPUDirect Storage.
 *
 * Falls back to cuFile compatibility mode (NVMe -> RAM -> VRAM)
 * if the nvidia-fs kernel module is not available.
 */

#include <cuda_runtime.h>
#include <cufile.h>

#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

// ─── Error checking helpers ───────────────────────────────────────────────

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            throw std::runtime_error(                                      \
                std::string("CUDA error in ") + #call + ": " +            \
                cudaGetErrorString(err));                                   \
        }                                                                  \
    } while (0)

#define CUFILE_CHECK(call, msg)                                            \
    do {                                                                   \
        CUfileError_t status = (call);                                     \
        if (status.err != CU_FILE_SUCCESS) {                               \
            throw std::runtime_error(                                      \
                std::string(msg) + ": cuFile error code " +                \
                std::to_string(status.err));                               \
        }                                                                  \
    } while (0)

// ─── GDS Context ──────────────────────────────────────────────────────────

struct GdsContext {
    bool initialized = false;
    bool driver_open = false;
};

static GdsContext g_ctx;

void gds_init() {
    if (g_ctx.initialized) return;

    CUfileError_t status = cuFileDriverOpen();
    if (status.err == CU_FILE_SUCCESS) {
        g_ctx.driver_open = true;
    } else {
        // Driver open failed — cuFile will work in compatibility mode
        // with individual file operations, just no driver-level features
        fprintf(stderr,
                "GdsLLM: cuFileDriverOpen returned %d — "
                "proceeding (compatibility mode likely)\n",
                status.err);
        g_ctx.driver_open = false;
    }
    g_ctx.initialized = true;
}

void gds_shutdown() {
    if (!g_ctx.initialized) return;
    if (g_ctx.driver_open) {
        cuFileDriverClose();
        g_ctx.driver_open = false;
    }
    g_ctx.initialized = false;
}

bool gds_is_driver_open() {
    return g_ctx.driver_open;
}

// ─── GDS Buffer ───────────────────────────────────────────────────────────

struct GdsBuffer {
    void* dev_ptr = nullptr;
    size_t size = 0;
    int device_id = 0;
    bool registered = false;
};

GdsBuffer* gds_alloc(size_t size, int device_id) {
    GdsBuffer* buf = new GdsBuffer();
    buf->device_id = device_id;

    // Round up to 4KB alignment
    size_t aligned_size = ((size + 4095) / 4096) * 4096;
    buf->size = aligned_size;

    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMalloc(&buf->dev_ptr, aligned_size));

    // Register buffer with cuFile for DMA
    CUfileError_t status = cuFileBufRegister(buf->dev_ptr, aligned_size, 0);
    if (status.err == CU_FILE_SUCCESS) {
        buf->registered = true;
    } else {
        // Registration failed — reads will still work in compat mode
        fprintf(stderr,
                "GdsLLM: cuFileBufRegister failed (code %d) — "
                "reads will use compatibility mode\n",
                status.err);
        buf->registered = false;
    }

    return buf;
}

void gds_free(GdsBuffer* buf) {
    if (!buf) return;
    if (buf->registered) {
        cuFileBufDeregister(buf->dev_ptr);
        buf->registered = false;
    }
    if (buf->dev_ptr) {
        cudaFree(buf->dev_ptr);
        buf->dev_ptr = nullptr;
    }
    delete buf;
}

// ─── GDS Read ─────────────────────────────────────────────────────────────

ssize_t gds_read(GdsBuffer* buf, const char* filepath,
                 size_t size, off_t file_offset, off_t buf_offset) {
    if (!buf || !buf->dev_ptr) {
        throw std::runtime_error("gds_read: invalid buffer");
    }
    if (buf_offset + size > buf->size) {
        throw std::runtime_error("gds_read: read would exceed buffer size");
    }

    // Open file with O_RDONLY | O_DIRECT for GDS
    int fd = open(filepath, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        // Fallback: try without O_DIRECT (some filesystems don't support it)
        fd = open(filepath, O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error(
                std::string("gds_read: cannot open file '") + filepath +
                "': " + strerror(errno));
        }
    }

    // Register file handle with cuFile
    CUfileDescr_t cf_descr;
    memset(&cf_descr, 0, sizeof(cf_descr));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileHandle_t cf_handle;
    CUfileError_t status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        close(fd);
        throw std::runtime_error(
            std::string("gds_read: cuFileHandleRegister failed for '") +
            filepath + "', error code " + std::to_string(status.err));
    }

    // Read file content directly into GPU buffer
    ssize_t bytes_read = cuFileRead(
        cf_handle,
        buf->dev_ptr,
        size,
        file_offset,
        buf_offset
    );

    // Cleanup
    cuFileHandleDeregister(cf_handle);
    close(fd);

    if (bytes_read < 0) {
        throw std::runtime_error(
            std::string("gds_read: cuFileRead failed for '") + filepath +
            "', error code " + std::to_string(bytes_read));
    }

    return bytes_read;
}
