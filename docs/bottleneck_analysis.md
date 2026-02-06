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

## Optimization Results

### Phase 1.1: Reusable scratch buffer (Bottleneck #6)

Pre-allocate a single fp16 buffer (448 MB for 70B) sized for the largest per-layer
weight. Reuse across all 7 dequant calls per layer instead of 511 torch::empty()/free
cycles per token.

**Result: 17% faster** — decode avg 7.689s → 6.356s/tok. Also eliminated timing
variance (6.3–11.0s → 6.3–6.5s). The allocation jitter was causing intermittent stalls.

### Phase 2: Double-buffering I/O (Bottleneck #5) — NO IMPROVEMENT

Implemented async prefetch (std::async bg thread for cuFileRead, buffer allocation on
main thread). Tested: 6.432s/tok vs 6.356s baseline — no measurable benefit.

**Why it doesn't help:** GDS reads already overlap naturally with GPU compute because:
1. `gds_read()` blocks the CPU but not the GPU (DMA transfer)
2. PyTorch kernel launches are async (return to CPU in µs)
3. While CPU blocks on `load_layer(N+1)`, GPU executes `transformer_block(N)` concurrently

The synchronous case effectively already achieves double-buffering at the hardware level.
Explicit async prefetch adds thread overhead for no benefit. **Reverted.**

### Phase 3: Fused dequant+GEMV (Bottleneck #1) — NO IMPROVEMENT

Implemented fused Q4_0 dequant+GEMV kernel: dequantizes blocks directly into dot-product
accumulators in registers, never materializing the fp16 intermediate. Reduces VRAM
bandwidth from ~2.8 GB/layer (write+read fp16) to ~0.35 GB/layer (read Q4_0 only) — 8x.

**Result: 6.465s/tok vs 6.36s baseline — no measurable benefit.**

**Why it doesn't help:** The GPU work is completely hidden behind NVMe I/O:
- NVMe read: 520 MB/layer × 73 layers / 7 GB/s = **5.4s** (dominates)
- Old GPU path (dequant+matmul): 2.8 GB × 73 / 717 GB/s = **0.29s** (hidden)
- Fused GPU path: 0.35 GB × 73 / 717 GB/s = **0.035s** (still hidden)

The GPU finishes each layer in ~4ms then waits ~70ms for the next NVMe read.
Making GPU work 8x faster just means it waits 66ms instead of 63ms.

**Code kept:** the fused kernel will matter when more layers are VRAM-cached (e.g.,
on a 24 GB GPU where 20+ layers fit), making the system GPU-bound instead of I/O-bound.

## The Real Bottleneck: NVMe Bandwidth

With 73 of 80 layers streamed from NVMe at 7 GB/s, the system is **NVMe-bandwidth-bound**.
All GPU-side optimizations (scratch buffer, fused kernel, double-buffering) are limited
because the GPU spends >90% of decode time idle, waiting for the next layer from NVMe.

**Breakdown of 6.4s decode:**
- NVMe reads (73 uncached layers): ~5.4s (84%)
- Cached layers compute (7 layers): ~0.6s (9%)
- Embed/norm/lm_head + overhead: ~0.4s (7%)

**Paths to further improvement:**
1. **More VRAM** — 24 GB GPU caches ~20 layers, 48 GB caches ~45 layers
2. **Faster NVMe** — Gen5 x4 (~14 GB/s) would halve I/O time
3. **Multiple NVMe** — RAID-0 across drives
4. **Smaller quant** — Q2_K halves weight data again (but may hurt quality)
5. **CPU RAM cache** — Stage layers in system RAM (50+ GB/s) before VRAM

## Summary (ranked by impact)

| Bottleneck | Est. impact | Fix | Status |
|---|---|---|---|
| **NVMe bandwidth (7 GB/s)** | **~84% of decode** | **More VRAM / faster NVMe** | **Hard limit** |
| ~~Dequant→matmul two-phase~~ | ~~15-20%~~ | ~~Fused dequant+GEMV~~ | **GPU already idle** |
| ~~Synchronous NVMe I/O~~ | ~~10%~~ | ~~Double-buffering~~ | **Natural overlap** |
| ~~Per-call fp16 allocation~~ | ~~3-5%~~ | ~~Reusable scratch buffer~~ | **Done: -17%** |

Current decode: **6.36s/tok** (from 7.69s baseline). The only Phase 1.1 scratch buffer
optimization had measurable impact because it eliminated CUDA allocator jitter. All
other GPU optimizations are masked by the 5.4s NVMe read bottleneck.
