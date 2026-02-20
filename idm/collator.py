from __future__ import annotations

from typing import Any

import numpy as np


class VideoSFTCollator:
    def __init__(
        self,
        processor: Any,
        instruction_text: str,
        video_fps: float | None = None,
        mask_no_op_actions: bool = False,
        mask_mouse_actions: bool = False,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.instruction_text = instruction_text
        self.video_fps = video_fps
        self.mask_no_op_actions = mask_no_op_actions
        self.mask_mouse_actions = mask_mouse_actions

    def _should_mask_action(self, action_s: str) -> bool:
        action_s = action_s.strip()
        if self.mask_no_op_actions and action_s == "NO_OP":
            return True
        if self.mask_mouse_actions and "MOUSE_" in action_s:
            return True
        return False

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

    def _token_count(self, text_s: str) -> int:
        tok_d = self.tokenizer(text_s, add_special_tokens=False)
        input_ids = tok_d.get("input_ids", [])
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)

    def _token_offsets(self, text_s: str) -> list[tuple[int, int]] | None:
        try:
            tok_d = self.tokenizer(
                text_s,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except TypeError:
            return None
        except Exception:
            return None

        offsets = tok_d.get("offset_mapping")
        if offsets is None:
            return None
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        try:
            return [(int(span[0]), int(span[1])) for span in offsets]
        except Exception:
            return None

    def _action_token_spans(self, target_s: str) -> list[tuple[str, int, int]]:
        spans_L = self._action_char_spans(target_s)
        offsets_L = self._token_offsets(target_s)
        if offsets_L is not None:
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

        return [
            (
                action_s,
                self._token_count(target_s[:start_char_i]),
                self._token_count(target_s[:end_char_i]),
            )
            for action_s, start_char_i, end_char_i in spans_L
        ]

    def _mask_mask_action_labels(
        self,
        labels_BS: Any,
        target_B: Any,
        prompt_lens_B: list[int],
    ) -> None:
        if not self.mask_no_op_actions and not self.mask_mouse_actions:
            return

        seq_len_i = int(labels_BS.shape[1])
        for b_i, (target_s, prompt_len_i) in enumerate(zip(target_B, prompt_lens_B)):
            for action_s, start_tok_i, end_tok_i in self._action_token_spans(
                str(target_s)
            ):
                if not self._should_mask_action(action_s):
                    continue
                start_i = int(prompt_len_i) + int(start_tok_i)
                end_i = int(prompt_len_i) + int(end_tok_i)
                start_i = max(start_i, 0)
                end_i = min(end_i, seq_len_i)
                if end_i > start_i:
                    labels_BS[b_i, start_i:end_i] = -100

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

    def _prompt_enc_d_from_frames(self, frames_BSHWC: np.ndarray) -> dict[str, Any]:
        videos_B = []
        prompt_text_B = []
        video_metadata_B = []
        for frames_SHWC in frames_BSHWC:
            prompt_msgs, _ = self._messages("")
            prompt_s = self.processor.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            videos_B.append(self._video_input(frames_SHWC))
            prompt_text_B.append(prompt_s)
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
        prompt_text_B = []
        full_text_B = []
        video_metadata_B = []
        for frames_SHWC, target_s in zip(frames_BSHWC, target_B):
            prompt_msgs, full_msgs = self._messages(target_s)
            prompt_s = self.processor.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_s = self.processor.apply_chat_template(
                full_msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            videos_B.append(self._video_input(frames_SHWC))
            prompt_text_B.append(prompt_s)
            full_text_B.append(full_s)
            if self.video_fps is not None:
                video_metadata_B.append(self._video_metadata(frames_SHWC))

        processor_kwargs = {
            "text": full_text_B,
            "videos": videos_B,
            "padding": True,
            "return_tensors": "pt",
        }
        prompt_kwargs = {
            "text": prompt_text_B,
            "videos": videos_B,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.video_fps is not None:
            processor_kwargs["video_metadata"] = video_metadata_B
            prompt_kwargs["video_metadata"] = video_metadata_B

        enc_d = self.processor(**processor_kwargs)
        enc_d.pop("token_type_ids", None)
        prompt_enc_d = self.processor(**prompt_kwargs)
        prompt_enc_d.pop("token_type_ids", None)

        prompt_mask_BS = prompt_enc_d["attention_mask"]
        prompt_lens_B = [int(x) for x in prompt_mask_BS.sum(dim=1).tolist()]

        input_ids_BS = enc_d["input_ids"]
        labels_BS = input_ids_BS.clone()

        # Prompt tokens and padding should not contribute to loss.
        for b_i, prompt_len in enumerate(prompt_lens_B):
            labels_BS[b_i, :prompt_len] = -100
        labels_BS[enc_d["attention_mask"] == 0] = -100
        self._mask_mask_action_labels(
            labels_BS=labels_BS,
            target_B=target_B,
            prompt_lens_B=prompt_lens_B,
        )

        enc_d["labels"] = labels_BS
        enc_d["prompt_lens"] = prompt_lens_B
        enc_d["videos"] = videos_B
        return enc_d
