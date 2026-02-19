from __future__ import annotations

from typing import Any

import numpy as np


class VideoSFTCollator:
    def __init__(
        self,
        processor: Any,
        instruction_text: str,
        video_fps: float | None = None,
    ):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("Processor must expose `tokenizer`.")
        self.instruction_text = instruction_text
        self.video_fps = video_fps

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
        return [frames_SHWC[i] for i in range(frames_SHWC.shape[0])]

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
                video_metadata_B.append(
                    {
                        "total_num_frames": int(frames_SHWC.shape[0]),
                        "fps": float(self.video_fps),
                        "frames_indices": list(range(int(frames_SHWC.shape[0]))),
                    }
                )

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

        if "attention_mask" in prompt_enc_d:
            prompt_mask_BS = prompt_enc_d["attention_mask"]
            if hasattr(prompt_mask_BS, "sum") and hasattr(prompt_mask_BS, "dim"):
                prompt_lens_B = [int(x) for x in prompt_mask_BS.sum(dim=1).tolist()]
            else:
                prompt_lens_B = [int(x) for x in prompt_mask_BS.sum(axis=1).tolist()]
        else:
            raise ValueError("Processor output is missing `attention_mask`; cannot mask prompt tokens reliably.")

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
