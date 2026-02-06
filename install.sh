#!/usr/bin/env bash
# GdsLLM Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/rscunha13/gdsllm/main/install.sh | bash
set -euo pipefail

GDSLLM_HOME="${GDSLLM_HOME:-$HOME/.gdsllm}"
GDSLLM_VENV="$GDSLLM_HOME/venv"
GDSLLM_REPO="https://github.com/rscunha13/gdsllm.git"
BIN_DIR="$HOME/.local/bin"

# ── Colors ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${NC}    $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC}  $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC}  $1"; }
info() { echo -e "  ${CYAN}[INFO]${NC}  $1"; }

# ── Banner ───────────────────────────────────────────────────────────────────

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║           GdsLLM Installer                   ║"
echo "  ║  NVMe → VRAM weight streaming for LLaMA      ║"
echo "  ╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Check prerequisites ─────────────────────────────────────────────────────

echo -e "${BOLD}Checking prerequisites...${NC}"
ERRORS=0

# Linux
if [[ "$(uname -s)" != "Linux" ]]; then
    fail "Linux required (found: $(uname -s))"
    ERRORS=$((ERRORS + 1))
else
    ok "Linux $(uname -r)"
fi

# NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU: $GPU_NAME (driver $DRIVER_VER)"
else
    fail "NVIDIA GPU not detected (nvidia-smi not found)"
    ERRORS=$((ERRORS + 1))
fi

# CUDA toolkit
CUDA_HOME="${CUDA_HOME:-}"
if [[ -z "$CUDA_HOME" ]]; then
    # Auto-detect
    if command -v nvcc &>/dev/null; then
        CUDA_HOME=$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")
    else
        for d in /usr/local/cuda-* /usr/local/cuda; do
            if [[ -d "$d/include" ]]; then
                CUDA_HOME="$d"
                break
            fi
        done
    fi
fi

if [[ -n "$CUDA_HOME" && -f "$CUDA_HOME/bin/nvcc" ]]; then
    CUDA_VER=$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    ok "CUDA $CUDA_VER ($CUDA_HOME)"
    export CUDA_HOME
else
    fail "CUDA toolkit not found (set CUDA_HOME env var)"
    ERRORS=$((ERRORS + 1))
fi

# cuFile (GPUDirect Storage)
CUFILE_FOUND=0
for libdir in "$CUDA_HOME/targets/$(uname -m)-linux/lib" "$CUDA_HOME/lib64" "/usr/lib"; do
    if [[ -f "$libdir/libcufile.so" || -f "$libdir/libcufile.so.0" ]]; then
        CUFILE_FOUND=1
        ok "cuFile library ($libdir)"
        break
    fi
done
if [[ "$CUFILE_FOUND" -eq 0 ]]; then
    fail "cuFile library not found (install nvidia-gds package)"
    ERRORS=$((ERRORS + 1))
fi

# nvidia-fs kernel module (GDS driver)
if lsmod 2>/dev/null | grep -q nvidia_fs; then
    ok "nvidia-fs kernel module loaded"
else
    warn "nvidia-fs kernel module not loaded (GDS will use compatibility mode)"
fi

# Python
PYTHON=""
for py in python3.12 python3.11 python3.10 python3; do
    if command -v "$py" &>/dev/null; then
        PY_VER=$("$py" --version 2>&1 | awk '{print $2}')
        PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 10 ]]; then
            PYTHON="$py"
            ok "Python $PY_VER ($py)"
            break
        fi
    fi
done
if [[ -z "$PYTHON" ]]; then
    fail "Python >= 3.10 required"
    ERRORS=$((ERRORS + 1))
fi

# g++
if command -v g++ &>/dev/null; then
    GCC_VER=$(g++ --version 2>/dev/null | head -1)
    ok "g++ found ($GCC_VER)"
else
    fail "g++ not found (install build-essential)"
    ERRORS=$((ERRORS + 1))
fi

# NVMe
if ls /dev/nvme* &>/dev/null; then
    NVME_COUNT=$(ls /dev/nvme[0-9]n[0-9] 2>/dev/null | wc -l)
    ok "NVMe: $NVME_COUNT device(s) found"
else
    warn "No NVMe devices found (GPUDirect Storage requires NVMe)"
fi

echo ""

# Abort if critical checks failed
if [[ "$ERRORS" -gt 0 ]]; then
    echo -e "${RED}${BOLD}$ERRORS prerequisite(s) failed. Fix the issues above and re-run.${NC}"
    exit 1
fi

# ── Install ──────────────────────────────────────────────────────────────────

echo -e "${BOLD}Installing GdsLLM...${NC}"

# Create directories
mkdir -p "$GDSLLM_HOME" "$BIN_DIR"

# Create venv
if [[ ! -d "$GDSLLM_VENV" ]]; then
    info "Creating virtual environment at $GDSLLM_VENV"
    "$PYTHON" -m venv "$GDSLLM_VENV"
else
    info "Using existing venv at $GDSLLM_VENV"
fi

# Install PyTorch first (needed for building CUDA extensions)
info "Installing PyTorch..."
"$GDSLLM_VENV/bin/pip" install --quiet torch

# Install GdsLLM from GitHub
info "Installing GdsLLM (compiling CUDA extensions, this may take a few minutes)..."
"$GDSLLM_VENV/bin/pip" install "git+${GDSLLM_REPO}"

# Symlink gdsllm command
ln -sf "$GDSLLM_VENV/bin/gdsllm" "$BIN_DIR/gdsllm"
ok "Installed gdsllm to $BIN_DIR/gdsllm"

# ── Post-install setup ───────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}Configuration...${NC}"

ENV_FILE="$GDSLLM_HOME/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    # Ask for config
    echo ""
    read -rp "  Model storage directory (NVMe path for converted models): " MODEL_ROOT
    read -rp "  HuggingFace cache directory (for downloads): " HF_CACHE
    read -rp "  HuggingFace token (from https://huggingface.co/settings/tokens): " HF_TOKEN

    MODEL_ROOT="${MODEL_ROOT:-$HOME/gdsllm_models}"
    HF_CACHE="${HF_CACHE:-$HOME/hf_cache}"

    cat > "$ENV_FILE" <<EOF
HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
GDSLLM_MODEL_ROOT=$MODEL_ROOT
GDSLLM_HF_CACHE=$HF_CACHE
EOF

    # Create directories
    mkdir -p "$MODEL_ROOT" "$HF_CACHE"
    ok "Config saved to $ENV_FILE"
else
    ok "Config already exists at $ENV_FILE"
fi

# ── PATH check ───────────────────────────────────────────────────────────────

if ! echo "$PATH" | grep -q "$BIN_DIR"; then
    echo ""
    warn "$BIN_DIR is not in your PATH"
    echo "  Add it to your shell profile:"
    echo "    echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
    echo "    source ~/.bashrc"
fi

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}${BOLD}GdsLLM installed successfully!${NC}"
echo ""
echo "  Quick start:"
echo "    gdsllm pull meta-llama/Llama-2-7b-hf    # Download & convert a model"
echo "    gdsllm list                              # List local models"
echo "    gdsllm run Llama-2-7b-hf                 # Chat in terminal"
echo "    gdsllm serve --model-dir Llama-2-7b-hf   # Start API server"
echo ""
