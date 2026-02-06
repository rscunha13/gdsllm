# Bottleneck Analysis: 70B Q4_0 at ~70% GPU Utilization

## Pipeline per decode step (single token)

For each of the 80 layers, the sequence is:
1. **NVMe read** (GDS) → VRAM buffer (~520 MB/layer Q4_0)
2. **Dequant kernel** — Q4_0 blocks → fp16 (per weight matrix, 7 times per layer)
3. **Compute** — matmuls, attention, MLP
4. Repeat for next layer

Only 7 of 80 layers are cached. The other 73 are streamed from NVMe every token.

## Bottleneck #1: Dequant-then-matmul serialization (biggest impact)

In `gdsllm/runtime/llama.py:129-136`, `dequant_linear()` does:
```python
w_fp16 = dequant_gguf(block_data)   # kernel 1: dequant → new fp16 tensor
output = F.linear(x, w_fp16)         # kernel 2: matmul on the fp16 copy
```

This is **two separate kernel launches** with an **fp16 intermediate allocation** between them. For each layer, this happens 7 times (q, k, v, o, gate, up, down projections). That's **560 kernel pairs per token** for the uncached layers.

The GPU is idle between kernel launches waiting for:
- `torch::empty()` to allocate the fp16 output buffer (~64 MB for a 4096x8192 weight)
- Python/C++ dispatch overhead between `dequant_gguf` returning and `F.linear` starting

A fused dequant+GEMM kernel would eliminate the intermediate allocation entirely and keep the GPU fed continuously. This alone likely accounts for most of the ~30% idle time.

## Bottleneck #2: Per-weight-matrix dequant calls (Python→C++ roundtrip overhead)

Each `dequant_linear()` call crosses the Python→pybind11→CUDA boundary separately. Per layer, that's 7 crossings × 73 uncached layers = **511 Python→C++ roundtrips per token**. Each roundtrip involves:
- pybind11 argument conversion
- `torch::empty()` allocation
- Kernel launch
- Return to Python

These are individually fast (~10-50μs each), but at 511 calls they add up to 5-25ms of overhead per token — not huge relative to the ~7s total, but it adds up.

## Bottleneck #3: Dequant kernel efficiency

The CUDA kernel in `gdsllm/runtime/gguf_dequant.cu:76-108` uses **one thread per block of 32 weights**. Each thread:
- Reads 18 bytes (1 scale + 16 packed bytes)
- Writes 32 fp16 values (64 bytes)
- Does 32 float multiply-adds

This is heavily **memory-bandwidth bound** with poor occupancy characteristics:
- Each thread does very little compute (32 FMAs) relative to the memory access
- No shared memory usage — every thread does independent global memory reads
- The write pattern (out[j] and out[j+16]) has a 32-byte stride, causing non-coalesced writes

A better approach would be: one warp (32 threads) per block, each thread handling one weight — giving coalesced reads and writes. Or better yet, skip dequant entirely with a fused kernel.

## Bottleneck #4: `torch.cuda.empty_cache()` calls in SimpleScheduler

In `gdsllm/runtime/scheduler.py:153-154` (SimpleScheduler), there's a `torch.cuda.empty_cache()` after every layer. This forces CUDA to synchronize and reclaim memory, stalling the pipeline. With CachedScheduler this doesn't happen for cached layers, but for the 73 uncached layers there's still the `release_temp()` + `empty_cache()` cycle.

## Bottleneck #5: NVMe I/O is synchronous and blocks compute

`gds_read()` in `gdsllm/runtime/gds_bindings.cpp:99-101` is a blocking call — the CPU waits for the full layer file to arrive in VRAM before returning to Python. There's no overlap between:
- Loading layer N+1 from NVMe
- Computing layer N on GPU

Double-buffering (load next layer while computing current) would hide most of the NVMe latency. At ~7 GB/s Gen4 NVMe and ~520 MB/layer, each layer read takes ~74ms. Over 73 uncached layers that's ~5.4s of pure I/O — a significant fraction of the ~7.7s decode time.

## Bottleneck #6: Repeated fp16 allocation/deallocation

`py_dequant_gguf()` in `gdsllm/runtime/gds_bindings.cpp:218-221` allocates a fresh fp16 tensor every call via `torch::empty()`. For a single layer's 7 projections, that's 7 allocations and deallocations of large tensors (up to ~64 MB each). CUDA's memory allocator is not free — each `cudaMalloc`/`cudaFree` pair has overhead. A pre-allocated scratch buffer reused across projections would eliminate this.

## Summary (ranked by impact)

| Bottleneck | Est. GPU idle | Fix |
|---|---|---|
| Dequant→matmul two-phase | ~15-20% | Fused dequant+GEMM kernel |
| Synchronous NVMe I/O | ~10% | Double-buffering / async prefetch |
| Per-call fp16 allocation | ~3-5% | Reusable scratch buffer |
| Dequant kernel efficiency | ~2-3% | Warp-level, coalesced writes |
| Python→C++ roundtrips | ~1-2% | Batch dequant per layer |

The ~70% GPU utilization is consistent with the GPU being starved by the two-phase dequant+matmul pattern and synchronous I/O. The dequant kernel itself is fast (memory-bound, tiny vs matmul time), but the serialization and allocation overhead between dequant and matmul keeps the GPU waiting.
