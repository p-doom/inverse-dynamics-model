from __future__ import annotations

from typing import Any

from PIL import Image
import torch


class ChatSFTCollator:
    """Collator for multi-turn image/action chat SFT.

    Uses the Qwen3-VL processor to tokenise conversations that contain
    local-path images.  Labels are created so that loss is only computed
    on **assistant**-turn content tokens (the action predictions).

    Expected message format (after role-swapping in the data loader)::

        [
            {"role": "user",      "content": [{"type": "image", "image": "/path.jpg"}]},
            {"role": "assistant", "content": [{"type": "text",  "text": "MOUSE:0,1,0"}]},
            ...
        ]
    """

    def __init__(self, processor: Any):
        self.processor = processor
        self.tokenizer = processor.tokenizer

        self._im_start_id: int = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id: int = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        nl_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        self._nl_id: int | None = nl_ids[0] if nl_ids else None

    # ------------------------------------------------------------------
    # Label creation
    # ------------------------------------------------------------------

    def _create_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mask everything except assistant-turn content (+ ``<|im_end|>``)."""
        labels = input_ids.clone()
        labels[:] = -100
        B, S = input_ids.shape

        for b in range(B):
            ids = input_ids[b].tolist()
            mask = attention_mask[b].tolist()
            i = 0
            while i < S:
                if mask[i] == 0 or ids[i] != self._im_start_id:
                    i += 1
                    continue

                # -- find the role token(s) between <|im_start|> and \n --
                role_start = i + 1
                role_end = role_start
                while role_end < S:
                    if self._nl_id is not None and ids[role_end] == self._nl_id:
                        break
                    if role_end - role_start > 10:
                        break
                    role_end += 1

                role_text = self.tokenizer.decode(ids[role_start:role_end]).strip()

                # -- locate the content tokens until <|im_end|> --
                content_start = min(role_end + 1, S)
                content_end = content_start
                while content_end < S and ids[content_end] != self._im_end_id:
                    content_end += 1

                if role_text == "assistant":
                    end = min(content_end + 1, S)
                    for j in range(content_start, end):
                        if mask[j] == 1:
                            labels[b, j] = ids[j]

                i = min(content_end + 1, S)

        labels[attention_mask == 0] = -100
        return labels

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
        """Load PIL images referenced in the conversation."""
        images: list[Image.Image] = []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    path = item.get("image", item.get("url", ""))
                    images.append(Image.open(path).convert("RGB"))
        return images

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        messages_list = [example["messages"] for example in batch]

        text_list = self.processor.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(text_list, str):
            text_list = [text_list]

        flat_images: list[Image.Image] = []
        for messages in messages_list:
            flat_images.extend(self._extract_images(messages))

        inputs = self.processor(
            text=text_list,
            images=flat_images if flat_images else None,
            return_tensors="pt",
            padding=True,
        )
        inputs.pop("token_type_ids", None)

        labels = self._create_labels(inputs["input_ids"], inputs["attention_mask"])
        inputs["labels"] = labels
        return dict(inputs)
