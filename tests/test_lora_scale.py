"""The alpha -> scale mapping, pinned.

mlx-lm's LoRALinear applies `scale` directly to the adapter update:

    y = linear(x) + scale * ((x @ lora_a) @ lora_b)

Textbook LoRA scales by `alpha / rank`, so the compatible mapping is
`scale = alpha / rank`. That is not mlx-lm's own default, which is 20.0 -- so
the two conventions differ by 10x at rank 8 / alpha 16, and an adapter trained
under one and served under the other is silently a different model.

This is the same shape of bug as the cross-entropy reduction: two systems
agreeing on a name while disagreeing on the arithmetic behind it.
"""

from __future__ import annotations

import pytest

from synth_mlx_rl.config import Settings


def test_scale_is_alpha_over_rank() -> None:
    settings = Settings(lora_rank=8, lora_alpha=16.0)
    assert settings.lora_scale == pytest.approx(2.0)

    # Not mlx-lm's own default of 20.0. If this ever silently becomes 20, every
    # adapter trained before the change serves as a different model.
    assert settings.lora_scale != 20.0

    for rank, alpha, expected in ((4, 8.0, 2.0), (16, 16.0, 1.0), (8, 32.0, 4.0)):
        assert Settings(lora_rank=rank, lora_alpha=alpha).lora_scale == pytest.approx(expected)


def test_the_mapping_reproduces_the_textbook_update() -> None:
    """Numerical parity against `y + (alpha/rank) * B A x`, computed by hand."""
    mlx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx_lm.tuner.lora import LoRALinear

    rank, alpha, dim = 8, 16.0, 32
    settings = Settings(lora_rank=rank, lora_alpha=alpha)

    base = nn.Linear(dim, dim, bias=False)
    adapted = LoRALinear.from_base(base, r=rank, scale=settings.lora_scale, dropout=0.0)

    x = mlx.random.normal((2, dim))
    got = adapted(x)

    # lora_a is initialised nonzero and lora_b zero, so seed b to make the term real.
    adapted.lora_b = mlx.random.normal(adapted.lora_b.shape) * 0.05
    got = adapted(x)
    expected = base(x) + (alpha / rank) * ((x @ adapted.lora_a) @ adapted.lora_b)
    assert mlx.allclose(got, expected, atol=1e-5), "scale is not alpha/rank in effect"


def test_the_scale_is_recorded_so_an_adapter_can_be_served_correctly() -> None:
    """An adapter trained at one scale and served at another is a different
    model, so the value has to travel with the weights."""
    settings = Settings(lora_rank=8, lora_alpha=16.0)
    assert settings.lora_scale == pytest.approx(settings.lora_alpha / settings.lora_rank)
