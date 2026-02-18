from __future__ import annotations

from typing import Any

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


class VideoSFTCollator:
    def __init__(
        self,
        processor: Any,
        instruction_text: str,
        convert_frames_to_pil: bool = True,
    ):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("Processor must expose `tokenizer`.")
        self.instruction_text = instruction_text
        self.convert_frames_to_pil = convert_frames_to_pil

    def _messages(self, target_s: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
        frames_L = [frames_SHWC[i] for i in range(frames_SHWC.shape[0])]
        if self.convert_frames_to_pil and Image is not None:
            return [Image.fromarray(frame_HWC) for frame_HWC in frames_L]
        return frames_L

    def __call__(self, batch_d: dict[str, Any]) -> dict[str, Any]:
        frames_BSHWC = batch_d["frames"]
        target_B = batch_d["target_text"]

        videos_B = []
        full_text_B = []
        prompt_lens_B = []
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
            prompt_len = len(
                self.tokenizer(prompt_s, add_special_tokens=False)["input_ids"]
            )
            videos_B.append(self._video_input(frames_SHWC))
            full_text_B.append(full_s)
            prompt_lens_B.append(prompt_len)

        enc_d = self.processor(
            text=full_text_B,
            videos=videos_B,
            padding=True,
            return_tensors="pt",
        )
        enc_d.pop("token_type_ids", None)

        input_ids_BS = enc_d["input_ids"]
        labels_BS = (
            input_ids_BS.clone()
            if hasattr(input_ids_BS, "clone")
            else np.array(input_ids_BS, copy=True)
        )

        # Prompt tokens and padding should not contribute to loss.
        for b_i, prompt_len in enumerate(prompt_lens_B):
            labels_BS[b_i, :prompt_len] = -100
        if "attention_mask" in enc_d:
            labels_BS[enc_d["attention_mask"] == 0] = -100
        else:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                labels_BS[input_ids_BS == pad_id] = -100

        for b_i in range(len(prompt_lens_B)):
            sup_n = (
                int((labels_BS[b_i] != -100).sum().item())
                if hasattr((labels_BS[b_i] != -100).sum(), "item")
                else int((labels_BS[b_i] != -100).sum())
            )
            if sup_n <= 0:
                raise ValueError(f"No supervised assistant tokens for sample {b_i}.")

        enc_d["labels"] = labels_BS
        enc_d["prompt_lens"] = prompt_lens_B
        enc_d["videos"] = videos_B
        return enc_d
