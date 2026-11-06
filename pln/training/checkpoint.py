"""Checkpoint save/load for MCN. Stores configs and weights only (loadable with weights_only=True)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch

from pln.algorithm.config import MCNConfig, TrainConfig
from pln.algorithm.interaction_model import MCN
from pln.errors import PLNInputError

FORMAT_VERSION = 1


def save_checkpoint(path: str, model: MCN, train_cfg: Optional[TrainConfig] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model_cfg": model.cfg.to_dict(),
        "train_cfg": None if train_cfg is None else train_cfg.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on POSIX: a crash never leaves a half-written checkpoint


def load_checkpoint(path: str, map_location: str = "cpu") -> Tuple[MCN, Optional[TrainConfig], Dict[str, Any]]:
    if not os.path.exists(path):
        raise PLNInputError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise PLNInputError(f"unsupported checkpoint format {payload.get('format_version')!r}")
    model = MCN(MCNConfig.from_dict(payload["model_cfg"]))
    model.load_state_dict(payload["state_dict"])
    tc = payload.get("train_cfg")
    return model, (None if tc is None else TrainConfig.from_dict(tc)), payload.get("extra", {})
