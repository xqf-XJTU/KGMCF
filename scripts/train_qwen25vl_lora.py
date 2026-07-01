#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and launch Qwen2.5-VL LoRA fine-tuning for KGMCF.")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--learning-rate", default="2e-5")
    parser.add_argument("--max-steps", type=int, default=3900)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and write a launch manifest without starting training.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_root = Path(args.model_root)
    dataset = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_root": str(model_root),
        "dataset": str(dataset),
        "output_dir": str(output_dir),
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "image_size": args.image_size,
        "gradient_checkpointing": args.gradient_checkpointing,
        "model_root_exists": model_root.exists(),
        "dataset_exists": dataset.exists(),
        "launch_mode": "dry_run" if args.dry_run else "external_trainer_required",
    }
    with (output_dir / "lora_training_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    print("Training manifest written. Connect this manifest to the local Qwen2.5-VL training stack before launching full fine-tuning.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
