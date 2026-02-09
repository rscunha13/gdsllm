# GdsLLM Architecture

## Overview

GdsLLM is an LLM inference runtime that streams model weights directly from NVMe to GPU VRAM using NVIDIA GPUDirect Storage (GDS). This bypasses CPU RAM entirely, enabling inference on models larger than available VRAM.

```
Traditional:  NVMe → CPU RAM → PCIe → VRAM   (double copy, CPU bottleneck)
GdsLLM:       NVMe → DMA → VRAM               (zero-copy, hardware DMA)
```

**Key insight**: By loading weights layer-by-layer via DMA and immediately computing, GdsLLM trades latency for the ability to run models that don't fit in VRAM. For models that do fit, all layers are cached and inference is GPU-bound.

---

## Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│  API REQUEST                                                  │
│  curl -d '{"messages":[...]}' http://localhost:8000/v1/...   │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  FastAPI Server  (server/app.py)                              │
│  ├─ BearerTokenMiddleware → validate GDSLLM_API_TOKEN        │
│  ├─ Route to /api/* (Ollama) or /v1/* (OpenAI)               │
│  └─ Acquire inference_lock (GPU serialization)               │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  InferenceEngine  (engine.py)                                 │
│  ├─ Tokenizer: apply chat template → input_ids               │
│  ├─ Allocate KV cache                                        │
│  └─ Two-phase generation:                                    │
│     ├─ Prefill: full prompt in one pass                      │
│     └─ Decode: one token per forward pass                    │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Scheduler  (runtime/scheduler.py)                            │
│  For each layer:                                              │
│  ├─ LayerCache: check VRAM cache (hit → skip load)           │
│  ├─ GDS load: cuFileRead() → NVMe DMA → VRAM buffer         │
│  ├─ Zero-copy: view_tensor() → torch.Tensor                  │
│  └─ Compute: transformer_block(h, weights, kv_cache)         │
│     ├─ RMSNorm → Attention (RoPE, GQA) → RMSNorm → MLP     │
│     ├─ Prefill: dequant → cuBLAS GEMM                        │
│     └─ Decode:  fused dequant+GEMV kernel                    │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Sampling  (engine.py::_sample)                               │
│  ├─ Repeat penalty on seen tokens                            │
│  ├─ Temperature scaling                                      │
│  ├─ Top-k filtering                                          │
│  ├─ Top-p (nucleus) filtering                                │
│  └─ Multinomial sampling → next token                        │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Streaming Response                                           │
│  ├─ Ollama: NDJSON lines (one per token)                     │
│  └─ OpenAI: SSE events (data: {...}\n\n)                     │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
gdsllm/
├── gdsllm/
│   ├── __init__.py
│   ├── engine.py                  # InferenceEngine — generation loop + sampling
│   ├── cli.py                     # CLI (serve, run, pull, list, show, rm, stop)
│   ├── runtime/
│   │   ├── llama.py               # LLaMA forward pass (attention, MLP, RoPE)
│   │   ├── scheduler.py           # SimpleScheduler & CachedScheduler
│   │   ├── kv_cache.py            # KV cache for autoregressive decoding
│   │   ├── torch_bridge.py        # ModelWeights & LayerCache (weight loading)
│   │   ├── gds_bindings.cpp       # pybind11 bindings (Python ↔ C++)
│   │   ├── gds_io.cu              # cuFile DMA I/O (NVMe → VRAM)
│   │   ├── gguf_dequant.cu/h      # Q4_0/Q8_0 dequantization kernels
│   │   └── fused_gemv.cu/h        # Fused dequant+GEMV kernels
│   ├── server/
│   │   ├── app.py                 # FastAPI app, auth middleware, lifecycle
│   │   ├── routes_api.py          # Ollama-style API (/api/*)
│   │   ├── routes_openai.py       # OpenAI-compatible API (/v1/*)
│   │   └── schemas.py             # Pydantic request/response models
│   └── tools/
│       ├── download_model.py      # HuggingFace Hub downloader
│       ├── convert_weights.py     # Safetensors → GdsLLM flat binaries
│       └── convert_gguf.py        # GGUF → GdsLLM flat binaries
├── examples/
│   └── inference_demo.py          # Standalone inference example
├── setup.py                       # CUDA auto-detection + cuFile build
├── install.sh                     # curl installer
└── docs/
    └── architecture.md            # This file
```

---

## CUDA/C++ Layer

### GDS I/O (`gds_io.cu`)

Low-level cuFile operations for DMA transfers between NVMe and VRAM.

- **`gds_init()` / `gds_shutdown()`** — Initialize/close cuFile driver
- **`gds_alloc(size)`** — Allocate 4KB-aligned CUDA buffer, register with cuFile
- **`gds_read(handle, buffer, size, offset)`** — `cuFileRead()` DMA transfer
- **Fallback**: Compatibility mode if `nvidia-fs` kernel module is unavailable

Buffer lifecycle:
```
cudaMalloc(4KB-aligned) → cuFileBufRegister() → cuFileRead() → cuFileBufDeregister() → cudaFree()
```

### Dequantization Kernels (`gguf_dequant.cu`)

GPU kernels for GGUF block-quantized weights:

| Format | Block Size | Storage | Dequant |
|--------|-----------|---------|---------|
| Q8_0 | 32 weights | 34 bytes (fp16 scale + 32 int8) | `out = scale * qs[i]` |
| Q4_0 | 32 weights | 18 bytes (fp16 scale + 16 uint8 nibbles) | `out = scale * (nibble - 8)` |

Each CUDA thread processes one block of 32 weights.

### Fused GEMV Kernels (`fused_gemv.cu`)

Combined dequantization + matrix-vector multiply for single-token decode:

1. Load input vector into shared memory (e.g. 16KB for hidden_size=8192)
2. Each thread dequantizes a subset of blocks + accumulates dot product
3. Block-level reduction via warp shuffle + shared memory
4. One CUDA block per output row

This eliminates the fp16 intermediate buffer (e.g. 134MB for 70B down_proj), giving **10x speedup** for GPU-bound decode.

### Python Bindings (`gds_bindings.cpp`)

pybind11 module exposing:
- `load_file(path, size)` → `GdsBufferHandle`
- `view_tensor(handle, shape, dtype, offset)` → `torch.Tensor` (zero-copy via `from_blob()`)
- `dequant_gguf(handle, shape, quant_type, offset, scratch)` → `torch.Tensor` (fp16)
- `fused_dequant_gemv(handle, input, shape, quant_type, offset)` → `torch.Tensor`

Zero-copy design: shared pointers with custom deleters keep GDS buffers alive while tensors reference them.

---

## Runtime Layer

### Weight Loading (`torch_bridge.py`)

**`ModelWeights`**: Parses `metadata.json`, loads weight files via GDS.
- `load_layer(idx)` → dict of tensors + buffer handle
- `load_special(name)` → dict of tensors for embed/norm/lm_head

**`LayerCache`**: Adaptive VRAM cache with LRU eviction.
- Uses actual free VRAM (not static budget) to decide caching
- `get_layer(idx, min_free)` — load from cache or GDS
- `preload(min_free)` — cache all layers upfront if VRAM allows
- `ensure_free_vram(bytes)` — evict LRU layers until threshold met

### LLaMA Forward Pass (`llama.py`)

Implements the LLaMA transformer architecture:

- **`rms_norm(x, weight, eps)`** — RMSNorm normalization
- **`precompute_freqs_cis(dim, seq_len, theta)`** — RoPE frequencies (with LLaMA 3.x scaling)
- **`apply_rotary_emb(xq, xk, freqs)`** — Apply rotary position embeddings
- **`dequant_linear(x, weight, ...)`** — Linear with on-GPU dequantization
  - Prefill (seq_len > 1): separate dequant → cuBLAS GEMM
  - Decode (seq_len = 1): fused dequant+GEMV kernel
- **`attention(h, weights, freqs, mask, kv_cache)`** — Multi-head attention with GQA support
- **`mlp(h, weights, ...)`** — SwiGLU MLP: `down(gate * silu(up))`
- **`transformer_block()`** — Full pre-norm layer: attn + MLP with residuals

### Scheduler (`scheduler.py`)

Two scheduling strategies:

**`SimpleScheduler`** — Minimal VRAM, load/free per layer:
```
embed_tokens → [layer_0 → layer_N] → final_norm → lm_head
  load→compute→free for each step
```

**`CachedScheduler`** — Adaptive VRAM caching:
- Keeps frequently used layers resident in VRAM
- Dynamically reserves VRAM for activations based on sequence length
- Evicts LRU layers when activation memory grows
- Re-caches when VRAM frees up

**`estimate_activation_vram(config, seq_len)`** — Estimates peak VRAM for activations:
- Prefill: O(S^2) attention scores
- Decode: O(S) with KV cache

### KV Cache (`kv_cache.py`)

Pre-allocated tensors: `[num_layers, batch, num_kv_heads, max_seq_len, head_dim]`

- `update(layer_idx, k, v)` — Store new K/V, return full cached history
- `advance(num_tokens)` — Increment position after all layers complete
- `reset()` — Clear for new sequence (no reallocation)

---

## Engine Layer (`engine.py`)

**`InferenceEngine`** — High-level generation API.

Lifecycle:
1. Load tokenizer from model directory
2. Create scheduler (Simple or Cached based on config)
3. Initialize GDS driver, preload weights if requested
4. Generate tokens via `generate()` or `generate_chat()`
5. Shutdown: exit GDS context, free VRAM

Two-phase generation:
- **Prefill**: Process entire prompt in one forward pass, populate KV cache
- **Decode**: One token per forward pass, reuse cached K/V

Sampling (`_sample()`):
1. Apply repeat penalty to tokens already in context
2. Scale logits by temperature
3. Top-k filtering (keep k highest logits)
4. Top-p nucleus filtering (keep smallest set with cumulative probability >= p)
5. Multinomial sampling from resulting distribution

Token events are yielded as `TokenEvent` dataclass with text, timing, and completion stats.

---

## Server Layer (`server/`)

### FastAPI Application (`app.py`)

- **Lifespan**: Load `InferenceEngine` on startup, shutdown on exit
- **Auth**: Optional `BearerTokenMiddleware` via `GDSLLM_API_TOKEN` env var
- **Concurrency**: `inference_lock` serializes GPU access (single request at a time)
- **CORS**: Open for local development

### Ollama-style API (`routes_api.py`)

| Endpoint | Description |
|----------|-------------|
| `POST /api/generate` | Text completion (NDJSON stream or JSON) |
| `POST /api/chat` | Chat completion (NDJSON stream or JSON) |
| `GET /api/tags` | List local models |
| `POST /api/show` | Model info and config |

### OpenAI-compatible API (`routes_openai.py`)

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completion (SSE stream or JSON) |
| `GET /v1/models` | List available models |

Compatible with the OpenAI Python SDK.

### Sampling Parameters

Both APIs accept:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.0 | Sampling temperature (0 = greedy) |
| `max_tokens` | 128 | Maximum tokens to generate |
| `top_p` | 1.0 | Nucleus sampling threshold |
| `top_k` | 0 | Top-k filtering (0 = disabled) |
| `repeat_penalty` | 1.0 | Repetition penalty (1.0 = disabled) |

---

## CLI (`cli.py`)

| Command | Description |
|---------|-------------|
| `gdsllm serve <model>` | Start the API server |
| `gdsllm run <model>` | Interactive chat in terminal |
| `gdsllm pull <hf-model>` | Download HF model and convert to GdsLLM format |
| `gdsllm list` | List local models |
| `gdsllm show <model>` | Show model details |
| `gdsllm rm <model>` | Delete a model |
| `gdsllm stop` | Stop the running server |

Server lifecycle managed via PID file at `~/.gdsllm/server.pid`.

---

## Tools

### Download (`download_model.py`)

Downloads from HuggingFace Hub via `snapshot_download()`. Filters to only safetensors, config, and tokenizer files.

### Convert Safetensors (`convert_weights.py`)

Converts HuggingFace safetensors to GdsLLM flat binaries:

1. Index all tensors from safetensors files
2. Auto-detect model config from tensor shapes
3. Write special files: `embed_tokens.bin`, `final_norm.bin`, `lm_head.bin`
4. Write layer files: `layer_000.bin` ... `layer_NNN.bin` (9 tensors each)
5. Generate `metadata.json` with tensor offsets and shapes
6. Copy tokenizer files
7. Optional INT8 per-channel quantization (`--quantize int8`)

Tensor order per layer:
```
input_layernorm.weight
self_attn.q_proj.weight
self_attn.k_proj.weight
self_attn.v_proj.weight
self_attn.o_proj.weight
post_attention_layernorm.weight
mlp.gate_proj.weight
mlp.up_proj.weight
mlp.down_proj.weight
```

All tensors are 4KB-aligned (required for `O_DIRECT` / GDS DMA).

### Convert GGUF (`convert_gguf.py`)

Converts GGUF files to GdsLLM format:

- Keeps Q4_0/Q8_0 blocks as raw bytes (dequantized on GPU)
- Dequantizes unsupported types (Q6_K, Q5_K) to fp16 on CPU
- Extracts tokenizer from GGUF metadata and builds HF-compatible `tokenizer.json`

---

## Model Format

### Directory Layout

```
model_dir/
├── metadata.json              # Model config + tensor index
├── embed_tokens.bin           # Embedding weights
├── final_norm.bin             # Final RMSNorm weights
├── lm_head.bin                # Output projection weights
├── layer_000.bin              # Transformer layer 0 (9 tensors)
├── layer_001.bin              # Transformer layer 1
├── ...
├── layer_NNN.bin
├── tokenizer.json
├── tokenizer_config.json
└── tokenizer.model            # SentencePiece (if applicable)
```

### metadata.json

```json
{
  "model": "llama",
  "dtype": "float16",
  "quantization": "q4_0",
  "alignment": 4096,
  "vocab_size": 128256,
  "hidden_size": 8192,
  "intermediate_size": 28672,
  "num_layers": 80,
  "num_heads": 64,
  "num_kv_heads": 8,
  "head_dim": 128,
  "rms_norm_eps": 1e-5,
  "rope_theta": 500000.0,
  "max_seq_len": 131072,
  "rope_scaling": { "type": "llama3", "factor": 8.0, ... },
  "files": {
    "embed_tokens": {
      "path": "embed_tokens.bin",
      "size_bytes": ...,
      "tensors": [{ "name": "...", "shape": [...], "dtype": "...", "offset": 0, "size_bytes": ... }]
    },
    "final_norm": { ... },
    "lm_head": { ... },
    "layers": [
      {
        "index": 0,
        "path": "layer_000.bin",
        "size_bytes": ...,
        "tensors": [
          { "name": "input_layernorm.weight", "shape": [8192], "dtype": "float16", "offset": 0, ... },
          { "name": "self_attn.q_proj.weight", "shape": [8192, 8192], "dtype": "gguf_q4_0", "offset": 4096, "quant": "q4_0", "block_size": 32, ... },
          ...
        ]
      }
    ]
  }
}
```

---

## GDS Integration

### Initialization

```
gds_init() → cuFileDriverOpen()
  └─ Checks driver status, may fall back to compat mode
```

### Buffer Lifecycle

```
1. cudaMalloc()           — 4KB-aligned GPU buffer
2. cuFileBufRegister()    — Register for DMA
3. open(path, O_DIRECT)   — Open file for direct I/O
4. cuFileHandleRegister() — Register file descriptor
5. cuFileRead()           — DMA: NVMe SSD → VRAM
6. cuFileHandleDeregister() + close()
7. view_tensor()          — torch.Tensor via from_blob() (zero-copy)
   └─ shared_ptr deleter keeps buffer alive while tensor exists
```

### Compatibility Mode

If `nvidia-fs` kernel module is not loaded, cuFile falls back to compatibility mode which routes through CPU memory. Performance is reduced but functionality is preserved.

---

## Caching Strategy

The `LayerCache` in `torch_bridge.py` implements adaptive VRAM caching:

1. **On layer request**: Check if layer is cached in VRAM
2. **Cache hit**: Return cached tensors (no I/O)
3. **Cache miss**: Load via GDS, cache if free VRAM exceeds `min_free` threshold
4. **VRAM pressure**: Evict LRU layers when activations need more memory
5. **Recovery**: Re-cache layers when VRAM frees up after sequence ends

The `min_free` threshold is computed by `estimate_activation_vram()` based on current sequence length, ensuring activations always have enough space.

Behavior varies by model size:
- **7B on 16GB VRAM**: All 32 layers cached → GPU-bound (59 tok/s)
- **70B on 16GB VRAM**: 6-7 layers cached → NVMe-bound (0.15 tok/s)

---

## Performance

| Model | Layers | Cached | Bottleneck | Speed |
|-------|--------|--------|-----------|-------|
| 7B Q4_0 | 32 | All | GPU compute | 59.3 tok/s |
| 70B Q4_0 | 80 | 6-7 | NVMe I/O (~7 GB/s) | 0.15 tok/s |

Hardware: RTX 4080 16GB, CUDA 12.8, NVMe Gen4 x4.

Key optimizations:
- **Scratch buffer reuse**: -17% latency for NVMe-bound decode
- **Fused dequant+GEMV**: 10x speedup for GPU-bound decode
- **Adaptive caching**: Maximizes resident layers within VRAM budget
