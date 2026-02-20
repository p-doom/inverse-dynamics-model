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
        self.tokenizer = processor.tokenizer
        self.instruction_text = instruction_text
        self.video_fps = video_fps

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

        enc_d["labels"] = labels_BS
        enc_d["prompt_lens"] = prompt_lens_B
        enc_d["videos"] = videos_B
        return enc_d
