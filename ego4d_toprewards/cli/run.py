"""Run whole-caption or subgoal reward experiments."""

import argparse
from pathlib import Path

from ego4d_toprewards.core import run


def main() -> None:
    p = argparse.ArgumentParser(description="TOPReward-style logits+progress reward on Ego4D")
    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--out", help="JSONL output (default: runs/<model>[_subgoals]/topreward.jsonl)")
    p.add_argument("--cache-dir", default=None, help="HF datasets cache directory (default: Hugging Face cache settings)")
    p.add_argument("--max-frames", type=int, default=12, help="max frames per clip fed to the model")
    p.add_argument("--num-prefixes", type=int, default=8, help="prefix points for the progress curve")
    p.add_argument("--plots-dir", default=None, help="plot directory (default: plots beside --out; '' to disable)")
    p.add_argument(
        "--split-subgoals",
        action="store_true",
        help="split each episode caption into its sub-actions and score/plot each subgoal independently over the full clip",
    )
    args = p.parse_args()
    if args.num_samples < 1 or args.max_frames < 2 or args.num_prefixes < 1:
        p.error("--num-samples and --num-prefixes must be positive; --max-frames must be at least 2")
    tag = Path(args.model.rstrip("/")).name if Path(args.model).exists() else args.model.replace("/", "_")
    out = Path(args.out) if args.out else Path("runs") / (tag + ("_subgoals" if args.split_subgoals else "")) / "topreward.jsonl"
    plots = str(out.parent / "plots") if args.plots_dir is None else args.plots_dir
    run(
        num_samples=args.num_samples,
        model_name=args.model,
        out_path=str(out),
        cache_dir=args.cache_dir,
        max_frames=args.max_frames,
        num_prefixes=args.num_prefixes,
        plots_dir=plots or None,
        split_subgoals=args.split_subgoals,
    )


if __name__ == "__main__":
    main()
