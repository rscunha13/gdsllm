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

    def load_layer(self, layer_index: int) -> Tuple[Dict[str, torch.Tensor], object]:
        """Load all tensors for a transformer layer via GDS.

        Returns:
            (tensors_dict, buffer_handle)
            The caller MUST keep buffer_handle alive while using the tensors.
        """
        layer = self.layers[layer_index]
        filepath = os.path.join(self.model_dir, layer.path)

        # Single I/O operation: load entire layer file
        handle = self._ext.load_file(filepath, layer.size_bytes)

        # Create tensor views
        tensors = {}
        for t in layer.tensors:
            tensors[t.name] = self._ext.view_tensor(
                handle, t.shape, t.dtype, t.offset
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
            tensors[t.name] = self._ext.view_tensor(
                handle, t.shape, t.dtype, t.offset
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
    """VRAM cache for model layers loaded via GDS.

    Keeps loaded layers resident in VRAM to avoid repeated NVMe reads.
    Automatically determines how many layers fit based on available VRAM.
    """

    def __init__(
        self,
        model_weights: ModelWeights,
        vram_reserve_mb: int = 1024,
        device: int = 0,
    ):
        self._weights = model_weights
        self._device = device
        self._layer_cache: Dict[int, Tuple[Dict[str, torch.Tensor], object]] = {}
        self._special_cache: Dict[str, Tuple[Dict[str, torch.Tensor], object]] = {}
        self._hits = 0
        self._misses = 0

        # Compute VRAM budget from actual free memory
        free_vram, _total = torch.cuda.mem_get_info(device)
        reserve = vram_reserve_mb * 1024 * 1024
        self._budget = max(0, free_vram - reserve)
        self._used = 0

    def _fits(self, size_bytes: int) -> bool:
        return self._used + size_bytes <= self._budget

    def get_layer(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a layer's tensors, loading from NVMe if not cached."""
        if idx in self._layer_cache:
            self._hits += 1
            return self._layer_cache[idx][0]

        self._misses += 1
        tensors, handle = self._weights.load_layer(idx)
        layer_size = self._weights.layers[idx].size_bytes

        if self._fits(layer_size):
            self._layer_cache[idx] = (tensors, handle)
            self._used += layer_size

        return tensors

    def get_special(self, name: str) -> Dict[str, torch.Tensor]:
        """Get special tensors (embed_tokens, final_norm, lm_head)."""
        if name in self._special_cache:
            self._hits += 1
            return self._special_cache[name][0]

        self._misses += 1
        tensors, handle = self._weights.load_special(name)
        meta = self._weights.special[name]
        file_size = meta.size_bytes

        if self._fits(file_size):
            self._special_cache[name] = (tensors, handle)
            self._used += file_size

        return tensors

    def preload(self):
        """Load all weights that fit into VRAM upfront."""
        # Specials first (small, always needed)
        for name in ("embed_tokens", "final_norm", "lm_head"):
            if self._weights.special.get(name) is not None:
                self.get_special(name)

        # Then layers in order
        for idx in range(self._weights.num_layers):
            layer_size = self._weights.layers[idx].size_bytes
            if not self._fits(layer_size):
                break
            self.get_layer(idx)

    def evict_all(self):
        """Free all cached layers from VRAM."""
        self._layer_cache.clear()
        self._special_cache.clear()
        self._used = 0
        torch.cuda.empty_cache()

    @property
    def stats(self) -> dict:
        return {
            "cached_layers": len(self._layer_cache),
            "total_layers": self._weights.num_layers,
            "cached_specials": len(self._special_cache),
            "vram_used_mb": self._used / (1024 * 1024),
            "vram_budget_mb": self._budget / (1024 * 1024),
            "hits": self._hits,
            "misses": self._misses,
        }
