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
