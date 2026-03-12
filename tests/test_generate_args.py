from __future__ import annotations

import numpy as np
import pytest

from idm.generate import (
    Args,
    DEFAULT_INSTRUCTION,
    DEFAULT_SGLANG_INSTRUCTION,
    EvalMetrics,
    IDMEvaluator,
    _effective_instruction_text,
    _frames_to_mp4_data_uri,
    _sample_record_indices_by_path,
    _validate_args,
)


def test_validate_args_requires_hf_model_path_for_hf_backend(tmp_path):
    args = Args(
        data_root=str(tmp_path),
        backend="hf",
        hf_model_path="",
    )
    with pytest.raises(ValueError):
        _validate_args(args)


def test_validate_args_requires_sglang_url_for_sglang_backend(tmp_path):
    args = Args(
        data_root=str(tmp_path),
        backend="sglang",
        sglang_url="",
    )
    with pytest.raises(ValueError):
        _validate_args(args)


def test_eval_metrics_per_action_accuracy_is_case_insensitive():
    metrics = EvalMetrics(
        all_predictions=["NO_OP", "KEY_DOWN:W"],
        all_ground_truth=["no_op", "key_down:w"],
    )
    per_action = metrics.get_per_action_accuracy()
    assert per_action["no_op"]["f1"] == pytest.approx(1.0)
    assert per_action["key_down:w"]["f1"] == pytest.approx(1.0)


def test_frames_to_mp4_data_uri_rejects_invalid_shape():
    with pytest.raises(ValueError):
        _frames_to_mp4_data_uri(np.zeros((2, 4, 4), dtype=np.uint8), fps=30.0)


def test_build_sglang_content_uses_video_url(monkeypatch):
    monkeypatch.setattr(
        "idm.generate._frames_to_mp4_data_uri",
        lambda frames, fps: "data:video/mp4;base64,AAAA",
    )
    evaluator = IDMEvaluator(
        backend="sglang",
        sglang_url="http://localhost:30000",
    )
    content = evaluator._build_sglang_content(np.zeros((2, 4, 4, 3), dtype=np.uint8))
    assert content[0]["type"] == "video_url"
    assert content[0]["video_url"]["url"] == "data:video/mp4;base64,AAAA"
    assert content[1] == {"type": "text", "text": evaluator.instruction_text}


def test_effective_instruction_text_uses_strict_default_for_sglang():
    assert (
        _effective_instruction_text("sglang", DEFAULT_INSTRUCTION)
        == DEFAULT_SGLANG_INSTRUCTION
    )


def test_effective_instruction_text_keeps_custom_instruction_for_sglang():
    custom_instruction = "custom instruction"
    assert (
        _effective_instruction_text("sglang", custom_instruction) == custom_instruction
    )


def test_validate_args_rejects_negative_sample_seed(tmp_path):
    args = Args(
        data_root=str(tmp_path),
        backend="sglang",
        sample_seed=-1,
    )
    with pytest.raises(ValueError):
        _validate_args(args)


def test_validate_args_requires_hf_backend_for_apples_mode(tmp_path):
    args = Args(
        data_root=str(tmp_path),
        eval_mode="apples_to_apples",
        backend="sglang",
    )
    with pytest.raises(ValueError, match="apples_to_apples requires --backend hf"):
        _validate_args(args)


def test_validate_args_rejects_max_videos_for_apples_mode(tmp_path):
    args = Args(
        data_root=str(tmp_path),
        eval_mode="apples_to_apples",
        backend="hf",
        hf_model_path="Qwen/Qwen3-VL-2B-Instruct",
        max_videos=4,
    )
    with pytest.raises(ValueError, match="--max-videos is not supported"):
        _validate_args(args)


def test_validate_args_rejects_missing_hf_model_state_pkl(tmp_path):
    args = Args(
        data_root=str(tmp_path),
        backend="hf",
        hf_model_path="Qwen/Qwen3-VL-2B-Instruct",
        hf_model_state_pkl=str(tmp_path / "missing.pkl"),
    )
    with pytest.raises(ValueError, match="--hf-model-state-pkl not found"):
        _validate_args(args)


def test_sample_record_indices_by_path_is_reproducible_for_seed():
    paths = ["a", "b", "c"]
    counts = [10, 20, 30]
    plan_1 = _sample_record_indices_by_path(paths, counts, sample_size=12, seed=7)
    plan_2 = _sample_record_indices_by_path(paths, counts, sample_size=12, seed=7)
    assert plan_1 == plan_2
    assert sum(len(v) for v in plan_1.values()) == 12


def test_sample_record_indices_by_path_is_not_first_chunk_only():
    paths = ["a", "b", "c"]
    counts = [10, 10, 10]
    plan = _sample_record_indices_by_path(paths, counts, sample_size=8, seed=0)
    sampled_paths = set(plan)
    # Ensure sampling spans beyond the first file chunk.
    assert "b" in sampled_paths or "c" in sampled_paths
