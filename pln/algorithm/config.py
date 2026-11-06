"""Configuration dataclasses for the Moebius Competency Network and its training.

See research/mathematical_derivation/derivation.md for the meaning of each field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

import yaml

from pln.errors import PLNConfigError

_VARIANTS = ("cp", "ds")
_OBJECTIVES = ("nnpu", "unknown_as_negative")
_NNPU_MODES = ("kiryo", "clip")


@dataclass
class MCNConfig:
    """Architecture configuration (derivation.md section 6)."""

    in_dim_skill: int
    in_dim_comp: int
    embed_dim: int = 32
    max_order: int = 3
    rank: int = 16
    variant: str = "cp"
    ds_hidden: int = 32
    use_gate: bool = True
    gate_bias: float = 0.0
    init_scale: float = 0.3
    n_max: int = 12

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.variant not in _VARIANTS:
            raise PLNConfigError(f"variant must be one of {_VARIANTS}, got {self.variant!r}")
        for name in ("in_dim_skill", "in_dim_comp", "embed_dim", "max_order", "rank", "ds_hidden", "n_max"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise PLNConfigError(f"{name} must be a positive integer, got {value!r}")
        if self.max_order > self.n_max:
            raise PLNConfigError("max_order cannot exceed n_max")
        if self.init_scale <= 0:
            raise PLNConfigError("init_scale must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MCNConfig":
        allowed = {f.name for f in fields(cls)}
        unknown = set(d) - allowed
        if unknown:
            raise PLNConfigError(f"unknown MCNConfig keys: {sorted(unknown)}")
        return cls(**d)


@dataclass
class TrainConfig:
    """Training configuration (derivation.md sections 11 and 12)."""

    seed: int = 0
    epochs: int = 60
    batch_size: int = 128
    lr: float = 3e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "auto"
    objective: str = "nnpu"
    nnpu_mode: str = "kiryo"
    nnpu_beta: float = 0.0
    nnpu_gamma: float = 1.0
    lambda_neg: float = 1.0
    kappa: float = 1e-4
    lambda_mono: float = 0.0

    def __post_init__(self) -> None:
        if self.objective not in _OBJECTIVES:
            raise PLNConfigError(f"objective must be one of {_OBJECTIVES}, got {self.objective!r}")
        if self.nnpu_mode not in _NNPU_MODES:
            raise PLNConfigError(f"nnpu_mode must be one of {_NNPU_MODES}, got {self.nnpu_mode!r}")
        if self.epochs < 1 or self.batch_size < 1:
            raise PLNConfigError("epochs and batch_size must be >= 1")
        if self.lr <= 0:
            raise PLNConfigError("lr must be positive")
        for name in ("lambda_neg", "kappa", "lambda_mono", "nnpu_beta", "nnpu_gamma", "weight_decay"):
            if getattr(self, name) < 0:
                raise PLNConfigError(f"{name} must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        allowed = {f.name for f in fields(cls)}
        unknown = set(d) - allowed
        if unknown:
            raise PLNConfigError(f"unknown TrainConfig keys: {sorted(unknown)}")
        return cls(**d)


def load_yaml(path: str, model_overrides: Optional[Dict[str, Any]] = None) -> "tuple[MCNConfig, TrainConfig]":
    """Load a YAML file with `model:` and `train:` sections."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    model = dict(raw.get("model", {}))
    model.update(model_overrides or {})
    return MCNConfig.from_dict(model), TrainConfig.from_dict(dict(raw.get("train", {})))
