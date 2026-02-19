from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from idm.eval_sampler import IDMEvaluator, SlidingWindowConfig, load_val_videos


@dataclass
class Args:
    data_root: str
    sglang_url: str = "http://localhost:30000"
    output_dir: str = "./eval_results"
    image_h: int = 90
    image_w: int = 160
    image_c: int = 3
    seq_len: int = 128
    max_videos: int | None = None
    visualize: bool = False
    wandb_project: str = "idm-eval"
    wandb_run_name: str | None = None


if __name__ == "__main__":
    import wandb
    args = tyro.cli(Args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "data_root": args.data_root,
            "seq_len": args.seq_len,
            "max_videos": args.max_videos,
            "image_h": args.image_h,
            "image_w": args.image_w,
        },
    )

    evaluator = IDMEvaluator(args.sglang_url, SlidingWindowConfig(seq_len=args.seq_len))
    data_iter = load_val_videos(args.data_root, args.image_h, args.image_w, args.image_c)
    metrics = evaluator.evaluate(
        data_iter, 
        args.max_videos, 
        visualize=args.visualize, 
        output_dir=output_dir,
        use_wandb=args.use_wandb,
    )

    print(f"\nAccuracy: {metrics.accuracy:.4f} ({metrics.correct}/{metrics.total})")
    
    per_action = metrics.get_per_action_accuracy()
    print("\nPer-Action Metrics:")
    print(f"{'Action':<20} {'Support':>8} {'Recall':>8} {'Precision':>8} {'F1':>8}")
    print("-" * 56)
    for action, stats in sorted(per_action.items(), key=lambda x: -x[1]["support"]):
        print(f"{action:<20} {stats['support']:>8} {stats['recall']:>8.3f} {stats['precision']:>8.3f} {stats['f1']:>8.3f}")

    wandb.finish()