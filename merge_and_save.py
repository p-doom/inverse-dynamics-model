#!/usr/bin/env python3
"""Merge a LoRA checkpoint into the base model and save for serving."""

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to step_N/checkpoint.pt"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Where to save merged model"
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--train-vision",
        action="store_true",
        help="Include ViT LoRA targets (qkv, attn.proj, linear_fc1, linear_fc2)",
    )
    args = parser.parse_args()

    print(f"Loading base model {args.model_id}...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    lora_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ]
    if args.train_vision:
        lora_targets.extend(["qkv", "attn.proj", "linear_fc1", "linear_fc2"])

    print(f"Applying LoRA config (targets={lora_targets})...")
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=lora_targets,
        ),
    )

    # torch.compile(patch_embed) adds _orig_mod prefix to state dict keys.
    # Apply the same compile so key names match, then remove it after loading.
    model.base_model.model.model.visual.patch_embed = torch.compile(
        model.base_model.model.model.visual.patch_embed
    )

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Step: {ckpt['global_step']}")

    # Remove torch.compile wrapper before merge
    compiled_pe = model.base_model.model.model.visual.patch_embed
    model.base_model.model.model.visual.patch_embed = compiled_pe._orig_mod

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
