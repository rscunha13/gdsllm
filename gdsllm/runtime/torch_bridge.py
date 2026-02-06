"""
GdsLLM — PyTorch Bridge

High-level interface for loading model weights from GDS-format files
into CUDA tensors using the gds_io_ext C++ extension.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class TensorMeta:
    name: str
    shape: List[int]
    dtype: str
    offset: int
    size_bytes: int
    padded_size: int
    quant: Optional[str] = None
    scale_shape: Optional[List[int]] = None
    scale_offset: Optional[int] = None
    scale_size_bytes: Optional[int] = None
    scale_padded_size: Optional[int] = None


@dataclass
class LayerMeta:
    index: int
    path: str
    size_bytes: int
    tensors: List[TensorMeta] = field(default_factory=list)


@dataclass
class FileMeta:
    path: str
    size_bytes: int
    tensors: List[TensorMeta] = field(default_factory=list)


def _parse_tensors(raw_list: list) -> List[TensorMeta]:
    return [
        TensorMeta(
            name=t["name"],
            shape=t["shape"],
            dtype=t["dtype"],
            offset=t["offset"],
            size_bytes=t["size_bytes"],
            padded_size=t["padded_size"],
            quant=t.get("quant"),
            scale_shape=t.get("scale_shape"),
            scale_offset=t.get("scale_offset"),
            scale_size_bytes=t.get("scale_size_bytes"),
            scale_padded_size=t.get("scale_padded_size"),
        )
        for t in raw_list
    ]


class ModelWeights:
    """Manages loading model weights from GDS-format .bin files."""

    def __init__(self, model_dir: str):
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.model_dir = model_dir
        self._gds_initialized = False

        # Parse layer metadata
        self.layers: List[LayerMeta] = []
        for lm in self.metadata["files"]["layers"]:
            self.layers.append(
                LayerMeta(
                    index=lm["index"],
                    path=lm["path"],
                    size_bytes=lm["size_bytes"],
                    tensors=_parse_tensors(lm["tensors"]),
                )
            )

        # Parse special file metadata
        self.special: Dict[str, Optional[FileMeta]] = {}
        for name in ("embed_tokens", "final_norm", "lm_head"):
            raw = self.metadata["files"].get(name)
            if raw is not None:
                self.special[name] = FileMeta(
                    path=raw["path"],
                    size_bytes=raw["size_bytes"],
                    tensors=_parse_tensors(raw["tensors"]),
                )
            else:
                self.special[name] = None

    def init_gds(self):
        from gdsllm.runtime import gds_io_ext
        self._ext = gds_io_ext
        self._ext.init()
        self._gds_initialized = True

    def shutdown_gds(self):
        if self._gds_initialized:
            self._ext.shutdown()
            self._gds_initialized = False

    def _make_tensor_entry(self, handle, t: TensorMeta):
        """Create a tensor view or GGUF deferred-dequant dict for a single tensor."""
        if t.quant in ("q4_0", "q8_0"):
            # GGUF block-quantized: store metadata dict for deferred GPU dequant
            return {
                "buffer": handle,
                "shape": t.shape,
                "gguf_type": t.quant,
                "offset": t.offset,
            }
        return self._ext.view_tensor(handle, t.shape, t.dtype, t.offset)

    def load_layer(self, layer_index: int) -> Tuple[Dict[str, torch.Tensor], object]:
        """Load all tensors for a transformer layer via GDS.

        Returns:
            (tensors_dict, buffer_handle)
            The caller MUST keep buffer_handle alive while using the tensors.
            For INT8 quantized weights, dict includes both "name" (int8) and
            "name_without_.weight" + ".scale" (fp16 scale).
            For GGUF block-quantized weights (Q4_0, Q8_0), dict values are
            metadata dicts with buffer/shape/type/offset for deferred dequant.
        """
        layer = self.layers[layer_index]
        filepath = os.path.join(self.model_dir, layer.path)

        # Single I/O operation: load entire layer file
        handle = self._ext.load_file(filepath, layer.size_bytes)

        # Create tensor views or GGUF deferred-dequant dicts
        tensors = {}
        for t in layer.tensors:
            tensors[t.name] = self._make_tensor_entry(handle, t)
            # For INT8 quantized tensors, also create a view for the scale
            if t.quant == "int8" and t.scale_offset is not None:
                # Derive scale key: "self_attn.q_proj.weight" -> "self_attn.q_proj.scale"
                scale_name = t.name.rsplit(".weight", 1)[0] + ".scale" if t.name.endswith(".weight") else t.name + ".scale"
                tensors[scale_name] = self._ext.view_tensor(
                    handle, t.scale_shape, "float16", t.scale_offset
                )

        return tensors, handle

    def load_special(self, name: str) -> Tuple[Dict[str, torch.Tensor], object]:
        """Load a special file (embed_tokens, final_norm, lm_head) via GDS.

        Returns:
            (tensors_dict, buffer_handle)
            The caller MUST keep buffer_handle alive while using the tensors.
        """
        meta = self.special.get(name)
        if meta is None:
            raise ValueError(f"Special file '{name}' not available")

        filepath = os.path.join(self.model_dir, meta.path)
        handle = self._ext.load_file(filepath, meta.size_bytes)

        tensors = {}
        for t in meta.tensors:
            tensors[t.name] = self._make_tensor_entry(handle, t)
            if t.quant == "int8" and t.scale_offset is not None:
                scale_name = t.name.rsplit(".weight", 1)[0] + ".scale" if t.name.endswith(".weight") else t.name + ".scale"
                tensors[scale_name] = self._ext.view_tensor(
                    handle, t.scale_shape, "float16", t.scale_offset
                )

        return tensors, handle

    def is_driver_open(self) -> bool:
        if not self._gds_initialized:
            return False
        return self._ext.is_driver_open()

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def config(self) -> dict:
        """Return model configuration from metadata."""
        exclude = {"files", "alignment"}
        return {k: v for k, v in self.metadata.items() if k not in exclude}


class LayerCache:
    """Adaptive VRAM cache for model layers loaded via GDS.

    Uses actual free VRAM (not a static budget) to decide whether to cache
    a newly loaded layer. Evicts least-recently-used layers when VRAM
    pressure increases (e.g., longer sequences need more activation memory).
    Re-caches layers automatically when VRAM frees up.
    """

    def __init__(self, model_weights: ModelWeights, device: int = 0):
        self._weights = model_weights
        self._device = device
        self._layer_cache: Dict[int, Tuple[Dict[str, torch.Tensor], object]] = {}
        self._special_cache: Dict[str, Tuple[Dict[str, torch.Tensor], object]] = {}
        self._lru_order: List[int] = []  # layer indices, most recent at end
        self._temp_handle = None  # keeps non-cached layer buffer alive
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._used = 0

    def _free_vram(self) -> int:
        """Return actual free VRAM in bytes."""
        free, _ = torch.cuda.mem_get_info(self._device)
        return free

    def _can_cache(self, size_bytes: int, min_free: int) -> bool:
        """Check if we can cache size_bytes and still keep min_free available."""
        return self._free_vram() >= size_bytes + min_free

    def _touch_lru(self, idx: int):
        """Move layer to most-recently-used position."""
        if idx in self._lru_order:
            self._lru_order.remove(idx)
        self._lru_order.append(idx)

    def _evict_lru_layer(self) -> bool:
        """Evict the least-recently-used cached layer. Returns True if evicted."""
        if not self._lru_order:
            return False
        victim = self._lru_order.pop(0)
        if victim in self._layer_cache:
            layer_size = self._weights.layers[victim].size_bytes
            del self._layer_cache[victim]
            self._used -= layer_size
            self._evictions += 1
            torch.cuda.empty_cache()
            return True
        return False

    def get_layer(self, idx: int, min_free: int = 0) -> Dict[str, torch.Tensor]:
        """Get a layer's tensors, loading from NVMe if not cached.

        Args:
            idx: Layer index.
            min_free: Minimum VRAM (bytes) to keep free after caching.
                      The layer is only cached if free VRAM >= layer_size + min_free.
                      Pass activation_need here to prevent cache from starving compute.
        """
        if idx in self._layer_cache:
            self._hits += 1
            self._touch_lru(idx)
            return self._layer_cache[idx][0]

        self._misses += 1

        # Load from NVMe, evicting on OOM from cudaMalloc
        while True:
            try:
                tensors, handle = self._weights.load_layer(idx)
                break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                if not self._evict_lru_layer():
                    raise

        # Cache if there's room while respecting min_free
        layer_size = self._weights.layers[idx].size_bytes
        if self._can_cache(layer_size, min_free):
            self._layer_cache[idx] = (tensors, handle)
            self._used += layer_size
            self._touch_lru(idx)
        else:
            # Keep handle alive so tensor views remain valid until next load
            self._temp_handle = handle

        return tensors

    def get_special(self, name: str, min_free: int = 0) -> Dict[str, torch.Tensor]:
        """Get special tensors (embed_tokens, final_norm, lm_head)."""
        if name in self._special_cache:
            self._hits += 1
            return self._special_cache[name][0]

        self._misses += 1

        while True:
            try:
                tensors, handle = self._weights.load_special(name)
                break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                if not self._evict_lru_layer():
                    raise

        meta = self._weights.special[name]
        file_size = meta.size_bytes
        if self._can_cache(file_size, min_free):
            self._special_cache[name] = (tensors, handle)
            self._used += file_size
        else:
            self._temp_handle = handle

        return tensors

    def preload(self, min_free: int = 0):
        """Load all weights that fit into VRAM upfront.

        Args:
            min_free: Keep at least this many bytes free after preloading.
        """
        # Specials first (small, always needed)
        for name in ("embed_tokens", "final_norm", "lm_head"):
            if self._weights.special.get(name) is not None:
                self.get_special(name, min_free=min_free)

        # Then layers in order
        for idx in range(self._weights.num_layers):
            layer_size = self._weights.layers[idx].size_bytes
            if not self._can_cache(layer_size, min_free):
                break
            self.get_layer(idx, min_free=min_free)

    def ensure_free_vram(self, needed_bytes: int):
        """Evict cached layers until at least needed_bytes of VRAM is free.

        Evicts highest-indexed layers first (they will be needed last
        in a sequential forward pass).
        """
        torch.cuda.empty_cache()
        while self._free_vram() < needed_bytes and self._layer_cache:
            victim = max(self._layer_cache.keys())
            self.evict_layer(victim)

    def evict_layer(self, idx: int) -> bool:
        """Evict a specific layer from cache. Returns True if evicted."""
        if idx not in self._layer_cache:
            return False
        layer_size = self._weights.layers[idx].size_bytes
        del self._layer_cache[idx]
        self._used -= layer_size
        if idx in self._lru_order:
            self._lru_order.remove(idx)
        self._evictions += 1
        torch.cuda.empty_cache()
        return True

    def release_temp(self):
        """Release the temporary handle for the last non-cached layer."""
        self._temp_handle = None
        torch.cuda.empty_cache()

    def evict_all(self):
        """Free all cached layers from VRAM."""
        self._layer_cache.clear()
        self._special_cache.clear()
        self._lru_order.clear()
        self._temp_handle = None
        self._used = 0
        torch.cuda.empty_cache()

    @property
    def stats(self) -> dict:
        return {
            "cached_layers": len(self._layer_cache),
            "total_layers": self._weights.num_layers,
            "cached_specials": len(self._special_cache),
            "vram_used_mb": self._used / (1024 * 1024),
            "vram_free_mb": self._free_vram() / (1024 * 1024),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }
