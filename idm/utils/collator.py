from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import torch

from idm.utils.actions import action_has_nonzero_mouse_b, action_is_no_op_b


class VideoSFTCollator:
    def __init__(
        self,
        processor: Any,
        instruction_text: str,
        video_fps: float | None = None,
        no_op_loss_weight: float = 1.0,
        mouse_loss_weight: float = 1.0,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.instruction_text = instruction_text
        self.video_fps = video_fps
        self.no_op_loss_weight = float(no_op_loss_weight)
        self.mouse_loss_weight = float(mouse_loss_weight)
        prompt_msgs, _ = self._messages("")
        self._prompt_text = self.processor.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        self._prompt_len_cache: dict[tuple[int, int, int, int], int] = {}

    def _action_loss_weight(self, action_s: str) -> float:
        action_s = action_s.strip()
        if action_is_no_op_b(action_s):
            return self.no_op_loss_weight
        if action_has_nonzero_mouse_b(action_s):
            return self.mouse_loss_weight
        return 1.0

    def _action_char_spans(self, target_s: str) -> list[tuple[str, int, int]]:
        spans_L: list[tuple[str, int, int]] = []
        offset_i = 0
        for line_s in target_s.splitlines(keepends=True):
            core_s = line_s[:-1] if line_s.endswith("\n") else line_s
            parts_L = core_s.split(":", 1)
            if core_s.startswith("Frame ") and len(parts_L) == 2:
                action_start_i = len(parts_L[0]) + 1
                while action_start_i < len(core_s) and core_s[action_start_i] == " ":
                    action_start_i += 1
                action_s = core_s[action_start_i:]
                spans_L.append(
                    (
                        action_s,
                        offset_i + action_start_i,
                        offset_i + len(core_s),
                    )
                )
            offset_i += len(line_s)
        return spans_L

    def _action_token_spans(self, target_s: str) -> list[tuple[str, int, int]]:
        spans_L = self._action_char_spans(target_s)
        tok_d = self.tokenizer(
            target_s,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets_L = [(int(span[0]), int(span[1])) for span in tok_d["offset_mapping"]]
        out_L: list[tuple[str, int, int]] = []
        for action_s, start_char_i, end_char_i in spans_L:
            token_idx_L = []
            for tok_i, (tok_start_i, tok_end_i) in enumerate(offsets_L):
                if tok_end_i <= start_char_i or tok_start_i >= end_char_i:
                    continue
                token_idx_L.append(tok_i)
            if token_idx_L:
                out_L.append((action_s, token_idx_L[0], token_idx_L[-1] + 1))
            else:
                out_L.append((action_s, 0, 0))
        return out_L

    def _apply_action_loss_weights(
        self,
        label_weights_BS: Any,
        labels_BS: Any,
        target_B: Any,
        prompt_lens_B: list[int],
    ) -> None:
        if self.no_op_loss_weight == 1.0 and self.mouse_loss_weight == 1.0:
            label_weights_BS[labels_BS == -100] = 0.0
            return

        seq_len_i = int(label_weights_BS.shape[1])
        for b_i, (target_s, prompt_len_i) in enumerate(zip(target_B, prompt_lens_B)):
            for action_s, start_tok_i, end_tok_i in self._action_token_spans(
                str(target_s)
            ):
                weight_f = self._action_loss_weight(action_s)
                if weight_f == 1.0:
                    continue
                start_i = int(prompt_len_i) + int(start_tok_i)
                end_i = int(prompt_len_i) + int(end_tok_i)
                start_i = max(start_i, 0)
                end_i = min(end_i, seq_len_i)
                if end_i > start_i:
                    label_weights_BS[b_i, start_i:end_i] = weight_f
        label_weights_BS[labels_BS == -100] = 0.0

    def _messages(
        self, target_s: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        prompt_msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": self.instruction_text},
                ],
            }
        ]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": target_s}]
        return prompt_msgs, full_msgs

    def _video_input(self, frames_SHWC: np.ndarray) -> list[Any]:
        return [frames_SHWC[i] for i in range(frames_SHWC.shape[0])]

    def _video_metadata(self, frames_SHWC: np.ndarray) -> dict[str, Any]:
        return {
            "total_num_frames": int(frames_SHWC.shape[0]),
            "fps": float(self.video_fps),
            "frames_indices": list(range(int(frames_SHWC.shape[0]))),
        }

    def _prompt_len_for_frames(self, frames_SHWC: np.ndarray) -> int:
        key_t = tuple(frames_SHWC.shape[:4])
        cached_len_i = self._prompt_len_cache.get(key_t)
        if cached_len_i is not None:
            return int(cached_len_i)

        prompt_kwargs = {
            "text": [self._prompt_text],
            "videos": [self._video_input(frames_SHWC)],
            "padding": True,
            "return_tensors": "pt",
        }
        if self.video_fps is not None:
            prompt_kwargs["video_metadata"] = [self._video_metadata(frames_SHWC)]

        prompt_enc_d = self.processor(**prompt_kwargs)
        prompt_enc_d.pop("token_type_ids", None)
        prompt_len_i = int(prompt_enc_d["attention_mask"].sum(dim=1).item())
        self._prompt_len_cache[key_t] = prompt_len_i
        return prompt_len_i

    def _prompt_lens_from_frames(self, frames_BSHWC: np.ndarray) -> list[int]:
        if len(frames_BSHWC) == 0:
            return []

        ref_shape_t = tuple(int(x) for x in frames_BSHWC[0].shape)
        same_shape_b = True
        for frames_SHWC in frames_BSHWC[1:]:
            if tuple(int(x) for x in frames_SHWC.shape) != ref_shape_t:
                same_shape_b = False
                break

        if same_shape_b:
            prompt_len_i = self._prompt_len_for_frames(frames_BSHWC[0])
            return [prompt_len_i] * int(len(frames_BSHWC))

        prompt_enc_d = self._prompt_enc_d_from_frames(frames_BSHWC)
        return [int(x) for x in prompt_enc_d["prompt_lens"]]

    def _prompt_enc_d_from_frames(self, frames_BSHWC: np.ndarray) -> dict[str, Any]:
        videos_B = []
        prompt_text_B = []
        video_metadata_B = []
        for frames_SHWC in frames_BSHWC:
            videos_B.append(self._video_input(frames_SHWC))
            prompt_text_B.append(self._prompt_text)
            if self.video_fps is not None:
                video_metadata_B.append(self._video_metadata(frames_SHWC))

        prompt_kwargs = {
            "text": prompt_text_B,
            "videos": videos_B,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.video_fps is not None:
            prompt_kwargs["video_metadata"] = video_metadata_B

        prompt_enc_d = self.processor(**prompt_kwargs)
        prompt_enc_d.pop("token_type_ids", None)
        prompt_mask_BS = prompt_enc_d["attention_mask"]
        prompt_lens_B = [int(x) for x in prompt_mask_BS.sum(dim=1).tolist()]

        prompt_enc_d["prompt_lens"] = prompt_lens_B
        prompt_enc_d["videos"] = videos_B
        return prompt_enc_d

    def prompt_model_inputs(self, batch_d: dict[str, Any]) -> dict[str, Any]:
        frames_BSHWC = batch_d["frames"]
        return self._prompt_enc_d_from_frames(frames_BSHWC)

    def __call__(self, batch_d: dict[str, Any]) -> dict[str, Any]:
        frames_BSHWC = batch_d["frames"]
        target_B = batch_d["target_text"]

        videos_B = []
        full_msgs_B = []
        video_metadata_B = []
        for frames_SHWC, target_s in zip(frames_BSHWC, target_B):
            _, full_msgs = self._messages(target_s)
            full_msgs_B.append(full_msgs)
            videos_B.append(self._video_input(frames_SHWC))
            if self.video_fps is not None:
                video_metadata_B.append(self._video_metadata(frames_SHWC))
        full_text_B = self.processor.apply_chat_template(
            full_msgs_B,
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(full_text_B, str):
            full_text_B = [full_text_B]

        processor_kwargs = {
            "text": full_text_B,
            "videos": videos_B,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.video_fps is not None:
            processor_kwargs["video_metadata"] = video_metadata_B

        enc_d = self.processor(**processor_kwargs)
        enc_d.pop("token_type_ids", None)
        prompt_lens_B = self._prompt_lens_from_frames(frames_BSHWC)

        input_ids_BS = enc_d["input_ids"]
        labels_BS = input_ids_BS.clone()

        # Prompt tokens and padding should not contribute to loss.
        for b_i, prompt_len in enumerate(prompt_lens_B):
            labels_BS[b_i, :prompt_len] = -100
        labels_BS[enc_d["attention_mask"] == 0] = -100
        label_weights_BS = labels_BS.new_ones(
            labels_BS.shape,
            dtype=torch.float32,
        )
        self._apply_action_loss_weights(
            label_weights_BS=label_weights_BS,
            labels_BS=labels_BS,
            target_B=target_B,
            prompt_lens_B=prompt_lens_B,
        )

        enc_d["labels"] = labels_BS
        enc_d["label_weights"] = label_weights_BS
        enc_d["prompt_lens"] = prompt_lens_B
        enc_d["videos"] = videos_B
        return enc_d


class CollatorPrefetchIterator:
    def __init__(self, raw_it: Any, collator: VideoSFTCollator):
        self._raw_it = raw_it
        self._collator = collator
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._next_fut: Future[tuple[bytes, dict[str, Any]]] | None = None
        self._last_state_b: bytes = self._raw_it.get_state()
        self._closed_b = False
        self._submit_next()

    def _load_one(self) -> tuple[bytes, dict[str, Any]]:
        raw_batch_d = next(self._raw_it)
        state_after_b = self._raw_it.get_state()
        model_batch_d = self._collator(raw_batch_d)
        return state_after_b, model_batch_d

    def _submit_next(self) -> None:
        if self._closed_b:
            return
        self._next_fut = self._pool.submit(self._load_one)

    def last_state(self) -> bytes:
        return self._last_state_b

    def close(self) -> None:
        if self._closed_b:
            return
        self._closed_b = True
        self._pool.shutdown(wait=True, cancel_futures=False)
        self._next_fut = None

    def __iter__(self) -> "CollatorPrefetchIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        if self._next_fut is None:
            raise StopIteration
        try:
            state_after_b, model_batch_d = self._next_fut.result()
        except StopIteration:
            self.close()
            raise
        except Exception:
            self.close()
            raise
        self._last_state_b = state_after_b
        self._submit_next()
        return model_batch_d
