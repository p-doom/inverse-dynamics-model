from __future__ import annotations

import numpy as np
import torch

from idm.collator import VideoSFTCollator


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [ord(c) for c in text]}


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(
        self, messages, tokenize: bool, add_generation_prompt: bool
    ):
        del tokenize, add_generation_prompt
        user_s = messages[0]["content"][1]["text"]
        if len(messages) == 1:
            return f"U:{user_s}\nA:"
        return f"U:{user_s}\nA:{messages[1]['content']}"

    def __call__(self, text, videos, padding, return_tensors, video_metadata=None):
        del videos, padding, return_tensors, video_metadata
        ids_LL = [[ord(c) for c in t] for t in text]
        max_len = max(len(x) for x in ids_LL)
        ids_BS = torch.zeros((len(ids_LL), max_len), dtype=torch.long)
        attn_BS = torch.zeros((len(ids_LL), max_len), dtype=torch.long)
        type_BS = torch.ones((len(ids_LL), max_len), dtype=torch.long)
        for b_i, ids_L in enumerate(ids_LL):
            ids_BS[b_i, : len(ids_L)] = torch.tensor(ids_L, dtype=torch.long)
            attn_BS[b_i, : len(ids_L)] = 1
        return {
            "input_ids": ids_BS,
            "attention_mask": attn_BS,
            "token_type_ids": type_BS,
        }


class _FakeVideoExpandingProcessor(_FakeProcessor):
    def apply_chat_template(
        self, messages, tokenize: bool, add_generation_prompt: bool
    ):
        del tokenize, add_generation_prompt
        user_s = messages[0]["content"][1]["text"]
        if len(messages) == 1:
            return f"U:[VIDEO]{user_s}\nA:"
        return f"U:[VIDEO]{user_s}\nA:{messages[1]['content']}"

    def __call__(self, text, videos, padding, return_tensors, video_metadata=None):
        del padding, return_tensors, video_metadata
        expanded_text = []
        for text_s, frames_L in zip(text, videos):
            expanded_text.append(text_s.replace("[VIDEO]", "<V>" * len(frames_L)))
        ids_LL = [[ord(c) for c in t] for t in expanded_text]
        max_len = max(len(x) for x in ids_LL)
        ids_BS = torch.zeros((len(ids_LL), max_len), dtype=torch.long)
        attn_BS = torch.zeros((len(ids_LL), max_len), dtype=torch.long)
        type_BS = torch.ones((len(ids_LL), max_len), dtype=torch.long)
        for b_i, ids_L in enumerate(ids_LL):
            ids_BS[b_i, : len(ids_L)] = torch.tensor(ids_L, dtype=torch.long)
            attn_BS[b_i, : len(ids_L)] = 1
        return {
            "input_ids": ids_BS,
            "attention_mask": attn_BS,
            "token_type_ids": type_BS,
        }


def _batch():
    frames0_SHWC = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    frames1_SHWC = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    return {
        "frames": np.stack([frames0_SHWC, frames1_SHWC], axis=0),
        "target_text": ["Frame 0: left", "Frame 0: right"],
    }


def _single_item_batch(frames_n: int, target_s: str):
    frames_SHWC = np.zeros((frames_n, 2, 2, 3), dtype=np.uint8)
    return {
        "frames": np.expand_dims(frames_SHWC, axis=0),
        "target_text": [target_s],
    }


def _action_slice(
    prompt_len_i: int,
    target_s: str,
    action_s: str,
) -> slice:
    start_i = target_s.index(action_s)
    end_i = start_i + len(action_s)
    return slice(prompt_len_i + start_i, prompt_len_i + end_i)


def test_collator_masks_prompt_and_padding_tokens():
    collator = VideoSFTCollator(
        processor=_FakeProcessor(), instruction_text="Predict actions."
    )
    out_d = collator(_batch())
    labels_BS = out_d["labels"]
    input_ids_BS = out_d["input_ids"]
    attn_BS = out_d["attention_mask"]
    assert "token_type_ids" not in out_d
    for b_i, prompt_len in enumerate(out_d["prompt_lens"]):
        assert torch.all(labels_BS[b_i, :prompt_len] == -100)
        pad_mask_S = attn_BS[b_i] == 0
        assert torch.all(labels_BS[b_i, pad_mask_S] == -100)
        supervised_mask_S = labels_BS[b_i] != -100
        assert bool(supervised_mask_S.any())
        assert torch.all(
            labels_BS[b_i, supervised_mask_S] == input_ids_BS[b_i, supervised_mask_S]
        )


def test_collator_preserves_batch_size():
    collator = VideoSFTCollator(
        processor=_FakeProcessor(), instruction_text="Predict actions."
    )
    out_d = collator(_batch())
    assert out_d["input_ids"].shape[0] == 2
    assert len(out_d["videos"]) == 2


def test_collator_prompt_len_tracks_video_expansion():
    processor = _FakeVideoExpandingProcessor()
    collator = VideoSFTCollator(
        processor=processor,
        instruction_text="Predict actions.",
        video_fps=10.0,
    )
    out_d = collator(_batch())

    prompt_msgs, _ = collator._messages("Frame 0: left")
    prompt_s = processor.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    naive_prompt_len = len(
        processor.tokenizer(prompt_s, add_special_tokens=False)["input_ids"]
    )
    assert out_d["prompt_lens"][0] > naive_prompt_len

    labels_S = out_d["labels"][0]
    supervised_idx = torch.nonzero(labels_S != -100, as_tuple=False).flatten()
    assert int(supervised_idx.numel()) > 0
    assert int(supervised_idx[0].item()) == out_d["prompt_lens"][0]


def test_collator_zero_weight_no_op_masks_only_no_op_actions():
    target_s = "Frame 0: NO_OP\nFrame 1: KEY_DOWN:W"
    collator = VideoSFTCollator(
        processor=_FakeProcessor(),
        instruction_text="Predict actions.",
        zero_weight_no_op_actions=True,
    )
    out_d = collator(_single_item_batch(frames_n=2, target_s=target_s))

    labels_S = out_d["labels"][0]
    input_ids_S = out_d["input_ids"][0]
    prompt_len_i = out_d["prompt_lens"][0]

    noop_slice = _action_slice(prompt_len_i, target_s, "NO_OP")
    key_slice = _action_slice(prompt_len_i, target_s, "KEY_DOWN:W")
    assert torch.all(labels_S[noop_slice] == -100)
    assert torch.all(labels_S[key_slice] == input_ids_S[key_slice])


def test_collator_zero_weight_mouse_masks_all_mouse_actions():
    target_s = (
        "Frame 0: KEY_DOWN:W + MOUSE_MOVE\n"
        "Frame 1: MOUSE_DOWN:Left\n"
        "Frame 2: MOUSE_SCROLL\n"
        "Frame 3: KEY_UP:W"
    )
    collator = VideoSFTCollator(
        processor=_FakeProcessor(),
        instruction_text="Predict actions.",
        zero_weight_mouse_actions=True,
    )
    out_d = collator(_single_item_batch(frames_n=4, target_s=target_s))

    labels_S = out_d["labels"][0]
    input_ids_S = out_d["input_ids"][0]
    prompt_len_i = out_d["prompt_lens"][0]

    mixed_slice = _action_slice(prompt_len_i, target_s, "KEY_DOWN:W + MOUSE_MOVE")
    down_slice = _action_slice(prompt_len_i, target_s, "MOUSE_DOWN:Left")
    scroll_slice = _action_slice(prompt_len_i, target_s, "MOUSE_SCROLL")
    key_slice = _action_slice(prompt_len_i, target_s, "KEY_UP:W")

    assert torch.all(labels_S[mixed_slice] == -100)
    assert torch.all(labels_S[down_slice] == -100)
    assert torch.all(labels_S[scroll_slice] == -100)
    assert torch.all(labels_S[key_slice] == input_ids_S[key_slice])
