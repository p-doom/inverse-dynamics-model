from __future__ import annotations

import numpy as np
import pytest

from idm.generate import (
    Args,
    _validate_args,
    build_visualization_payload,
)


def test_validate_args_requires_hf_model_path(tmp_path):
    args = Args(
        data_root=str(tmp_path),
        hf_model_path="",
    )
    with pytest.raises(ValueError, match="--hf-model-path cannot be empty"):
        _validate_args(args)


def test_validate_args_rejects_missing_hf_model_state_pkl(tmp_path):
    args = Args(
        data_root=str(tmp_path),
        hf_model_state_pkl=str(tmp_path / "missing.pkl"),
    )
    with pytest.raises(ValueError, match="--hf-model-state-pkl not found"):
        _validate_args(args)


def test_validate_args_rejects_invalid_val_steps(tmp_path):
    args = Args(
        data_root=str(tmp_path),
        val_steps=0,
    )
    with pytest.raises(ValueError, match="--val-steps must be >= 1"):
        _validate_args(args)


def test_validate_args_rejects_invalid_skip_val_batches(tmp_path):
    args = Args(
        data_root=str(tmp_path),
        skip_val_batches=-1,
    )
    with pytest.raises(ValueError, match="--skip-val-batches must be >= 0"):
        _validate_args(args)


def test_build_visualization_payload_marks_all_frames_selected():
    frames = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    gt_actions = ["NO_OP", "KEY_DOWN:W", "MOUSE:1,0,0"]
    pred_actions = ["NO_OP", "KEY_DOWN:W", "NO_OP"]
    payload_d = build_visualization_payload(
        video_id="sample_000001",
        frames=frames,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        correct=2,
        total=3,
    )
    assert payload_d["video_id"] == "sample_000001"
    assert payload_d["action_count"] == 3
    assert payload_d["windows"][0]["start"] == 0
    assert payload_d["windows"][0]["use_end"] == 3
    assert all(frame_d["selected_by_stitching"] for frame_d in payload_d["frames"])
