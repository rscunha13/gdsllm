# GdsLLM

LLM inference runtime that streams weights directly from NVMe to VRAM via [GPUDirect Storage](https://developer.nvidia.com/gpudirect-storage), bypassing CPU RAM entirely. This enables running models larger than VRAM by loading layers on-demand through DMA.

```
NVMe SSD ──DMA──▶ GPU VRAM    (GdsLLM: ~7 GB/s, zero CPU copies)
NVMe SSD ──▶ Page Cache ──▶ CPU RAM ──PCIe──▶ GPU VRAM    (traditional)
```

## Performance

| Model | Quantization | Layers Cached | Throughput | Bottleneck |
|-------|-------------|---------------|------------|------------|
| LLaMA 7B | Q4_0 | All 32 (3.6 GB) | **59 tok/s** (17ms/tok) | GPU compute |
| LLaMA 70B | Q4_0 | 6 of 80 | 0.15 tok/s (6.5s/tok) | NVMe bandwidth |

*Hardware: RTX 4080 16GB, NVMe Gen4 x4 (~7 GB/s), CUDA 12.8*

## Requirements

- **Linux** (x86_64)
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit 12.x+** with cuFile (GPUDirect Storage)
- **NVMe SSD** (required for GDS DMA transfers)
- **Python 3.10+**

## Install

### Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/rscunha13/gdsllm/main/install.sh | bash
```

The installer checks prerequisites (GPU, CUDA, cuFile, NVMe, Python), creates a virtual environment, compiles CUDA extensions, and sets up the `gdsllm` command.

### Developer Install

```bash
git clone https://github.com/rscunha13/gdsllm.git
cd gdsllm
python -m venv .venv && source .venv/bin/activate
pip install torch
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your paths and HuggingFace token
```

Set `CUDA_HOME` if your CUDA toolkit isn't auto-detected:
```bash
export CUDA_HOME=/usr/local/cuda-12.8
```

## Quick Start

```bash
# Download and convert a model from HuggingFace
gdsllm pull meta-llama/Llama-2-7b-hf

# List local models
gdsllm list

# Chat in the terminal
gdsllm run Llama-2-7b-hf

# Start API server (Ollama + OpenAI compatible)
gdsllm serve --model-dir Llama-2-7b-hf --preload
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `gdsllm pull <model>` | Download HF model and convert to GdsLLM format |
| `gdsllm run <model>` | Interactive chat in terminal |
| `gdsllm serve` | Start the API server |
| `gdsllm stop` | Stop the running server |
| `gdsllm list` | List available local models |
| `gdsllm show <model>` | Show model details |
| `gdsllm rm <model>` | Delete a local model |

## API Endpoints

GdsLLM exposes both Ollama-style and OpenAI-compatible APIs on the same server.

### Ollama-style

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Text completion (streaming NDJSON) |
| `/api/chat` | POST | Chat completion (streaming NDJSON) |
| `/api/tags` | GET | List local models |
| `/api/show` | POST | Model details |

### OpenAI-compatible

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (SSE streaming) |
| `/v1/models` | GET | List models |

Works with the OpenAI Python SDK:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GDSLLM_MODEL_ROOT` | Root directory for converted GdsLLM models |
| `GDSLLM_HF_CACHE` | Directory for HuggingFace model downloads |
| `HUGGINGFACE_HUB_TOKEN` | HuggingFace authentication token |

Set these in a `.env` file in the project root (see `.env.example`), or export them in your shell.

## Architecture

```
gdsllm/
├── engine.py              # InferenceEngine — tokenizer + scheduler + generation loop
├── cli.py                 # CLI entry point (serve, run, pull, rm, list, show, stop)
├── server/
│   ├── app.py             # FastAPI app with model lifecycle management
│   ├── routes_api.py      # Ollama-compatible endpoints
│   ├── routes_openai.py   # OpenAI-compatible endpoints
│   └── schemas.py         # Pydantic request/response models
├── runtime/
│   ├── gds_io.cu          # cuFile DMA reads (NVMe → VRAM)
│   ├── gguf_dequant.cu    # Q4_0 / Q8_0 GPU dequantization kernels
│   ├── fused_gemv.cu      # Fused dequant + matrix-vector multiply
│   ├── scheduler.py       # Layer residency scheduler (cached vs. streaming)
│   ├── llama.py           # LLaMA forward pass
│   └── kv_cache.py        # KV cache for autoregressive decoding
└── tools/
    ├── download_model.py   # HuggingFace model downloader
    ├── convert_weights.py  # Safetensors → GdsLLM converter
    └── convert_gguf.py     # GGUF → GdsLLM converter
```

## License

MIT
