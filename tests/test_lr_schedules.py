import math

import pytest

from idm.utils.lr_schedules import LRScheduleArgs, lr_at_step


def test_cos_warmup_boundaries():
    cfg = LRScheduleArgs(
        schedule="cos",
        init_lr=0.0,
        max_lr=1.0,
        decay_end=0.1,
        max_steps=100,
        warmup_steps=10,
        wsd_decay_steps=20,
    )
    assert lr_at_step(cfg, 0) == pytest.approx(0.0)
    assert lr_at_step(cfg, 10) == pytest.approx(1.0)


def test_cos_decays_to_end_value():
    cfg = LRScheduleArgs(
        schedule="cos",
        init_lr=0.0,
        max_lr=1.0,
        decay_end=0.1,
        max_steps=100,
        warmup_steps=10,
        wsd_decay_steps=20,
    )
    assert lr_at_step(cfg, 100) == pytest.approx(0.1)


def test_wsd_has_warmup_plateau_and_decay():
    cfg = LRScheduleArgs(
        schedule="wsd",
        init_lr=0.0,
        max_lr=1.0,
        decay_end=0.2,
        max_steps=100,
        warmup_steps=10,
        wsd_decay_steps=20,
    )
    assert lr_at_step(cfg, 0) == pytest.approx(0.0)
    assert lr_at_step(cfg, 10) == pytest.approx(1.0)
    assert lr_at_step(cfg, 50) == pytest.approx(1.0)
    assert lr_at_step(cfg, 100) == pytest.approx(0.2)


def test_const_warmup_then_constant():
    cfg = LRScheduleArgs(
        schedule="const",
        init_lr=0.0,
        max_lr=1.0,
        decay_end=0.0,
        max_steps=100,
        warmup_steps=10,
        wsd_decay_steps=20,
    )
    assert lr_at_step(cfg, 0) == pytest.approx(0.0)
    assert lr_at_step(cfg, 10) == pytest.approx(1.0)
    assert lr_at_step(cfg, 99) == pytest.approx(1.0)
    assert lr_at_step(cfg, 100) == pytest.approx(1.0)


def test_invalid_schedule_raises():
    cfg = LRScheduleArgs(
        schedule="bad",
        init_lr=0.0,
        max_lr=1.0,
        decay_end=0.0,
        max_steps=100,
        warmup_steps=10,
        wsd_decay_steps=20,
    )
    with pytest.raises(ValueError):
        lr_at_step(cfg, 0)


def test_wsd_invalid_window_raises():
    cfg = LRScheduleArgs(
        schedule="wsd",
        init_lr=0.0,
        max_lr=1.0,
        decay_end=0.0,
        max_steps=10,
        warmup_steps=8,
        wsd_decay_steps=4,
    )
    with pytest.raises(ValueError):
        lr_at_step(cfg, 0)


def test_cos_invalid_warmup_raises():
    cfg = LRScheduleArgs(
        schedule="cos",
        init_lr=0.0,
        max_lr=1.0,
        decay_end=0.0,
        max_steps=10,
        warmup_steps=11,
        wsd_decay_steps=4,
    )
    with pytest.raises(ValueError):
        lr_at_step(cfg, 0)


def test_monotonic_warmup():
    cfg = LRScheduleArgs(
        schedule="const",
        init_lr=0.1,
        max_lr=0.5,
        decay_end=0.0,
        max_steps=50,
        warmup_steps=5,
        wsd_decay_steps=10,
    )
    vals = [lr_at_step(cfg, s) for s in range(6)]
    assert vals == sorted(vals)
    assert math.isclose(vals[-1], 0.5)
