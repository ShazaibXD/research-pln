import itertools

import numpy as np
import pytest
import torch

from pln.algorithm.config import MCNConfig
from pln.algorithm.interaction_model import MCN, Coeffs, OrderCoeffs


def make_model(variant="cp", k=3, n_max=6, seed=0, gate=True):
    torch.manual_seed(seed)
    cfg = MCNConfig(in_dim_skill=6, in_dim_comp=5, embed_dim=8, max_order=k, rank=4, variant=variant,
                    ds_hidden=8, n_max=n_max, use_gate=gate, init_scale=1.0)
    return MCN(cfg).double()


@pytest.fixture(params=["cp", "ds"])
def variant(request):
    return request.param


@pytest.fixture
def world():
    g = torch.Generator().manual_seed(123)
    return {"skill_feats": torch.randn(20, 6, generator=g, dtype=torch.float64),
            "comp_feats": torch.randn(4, 5, generator=g, dtype=torch.float64)}


@pytest.fixture
def sets():
    idx = torch.tensor([[1, 5, 7, 0, 0], [2, 3, 4, 9, 11], [13, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]).bool()
    return idx, mask


def coeffs_from_dict(mdict, bias, L, Mc=1):
    """Hand-built Coeffs for B=1 from {tuple_of_positions: value}; used to test explanations on known functions."""
    by_order = {}
    for T, v in mdict.items():
        by_order.setdefault(len(T), {})[tuple(sorted(T))] = v
    orders = []
    for j in range(1, max(by_order) + 1):
        combos = list(itertools.combinations(range(L), j))
        m = torch.zeros(1, len(combos), Mc, dtype=torch.float64)
        for n, T in enumerate(combos):
            if T in by_order.get(j, {}):
                m[0, n, 0] = by_order[j][T]
        orders.append(OrderCoeffs(j, torch.tensor(combos).reshape(len(combos), j),
                                  torch.ones(1, len(combos), dtype=torch.bool), m))
    return Coeffs(bias=torch.tensor([bias], dtype=torch.float64), orders=orders, length=L, comp_idx=torch.arange(Mc))
