from __future__ import annotations

import numpy as np

from idm.collator import VideoSFTCollator


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [ord(c) for c in text]}


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool):
        del tokenize, add_generation_prompt
        user_s = messages[0]["content"][1]["text"]
        if len(messages) == 1:
            return f"U:{user_s}\nA:"
        return f"U:{user_s}\nA:{messages[1]['content']}"

    def __call__(self, text, videos, padding, return_tensors):
        del videos, padding, return_tensors
        ids_LL = [[ord(c) for c in t] for t in text]
        max_len = max(len(x) for x in ids_LL)
        ids_BS = np.zeros((len(ids_LL), max_len), dtype=np.int64)
        attn_BS = np.zeros((len(ids_LL), max_len), dtype=np.int64)
        type_BS = np.ones((len(ids_LL), max_len), dtype=np.int64)
        for b_i, ids_L in enumerate(ids_LL):
            ids_BS[b_i, : len(ids_L)] = ids_L
            attn_BS[b_i, : len(ids_L)] = 1
        return {"input_ids": ids_BS, "attention_mask": attn_BS, "token_type_ids": type_BS}


def _batch():
    frames0_SHWC = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    frames1_SHWC = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    return {
        "frames": np.stack([frames0_SHWC, frames1_SHWC], axis=0),
        "target_text": ["Frame 0: left", "Frame 0: right"],
        "meta": [{"record_key": "k0"}, {"record_key": "k1"}],
    }


def test_collator_masks_prompt_and_padding_tokens():
    collator = VideoSFTCollator(processor=_FakeProcessor(), instruction_text="Predict actions.")
    out_d = collator(_batch())
    labels_BS = out_d["labels"]
    input_ids_BS = out_d["input_ids"]
    attn_BS = out_d["attention_mask"]
    assert "token_type_ids" not in out_d
    for b_i, prompt_len in enumerate(out_d["prompt_lens"]):
        assert np.all(labels_BS[b_i, :prompt_len] == -100)
        pad_mask_S = attn_BS[b_i] == 0
        assert np.all(labels_BS[b_i, pad_mask_S] == -100)
        supervised_mask_S = labels_BS[b_i] != -100
        assert supervised_mask_S.any()
        assert np.all(labels_BS[b_i, supervised_mask_S] == input_ids_BS[b_i, supervised_mask_S])


def test_collator_preserves_batch_size():
    collator = VideoSFTCollator(processor=_FakeProcessor(), instruction_text="Predict actions.")
    out_d = collator(_batch())
    assert out_d["input_ids"].shape[0] == 2
    assert len(out_d["videos"]) == 2
