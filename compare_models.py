"""compare_models.py — aggregate topreward_test.py runs across models.

Reads each model's `runs/<TAG>/topreward.jsonl`, then writes:
  * runs/comparison.json            — per-model VOC / reward / P(True) stats
  * runs/comparison_voc.png         — VOC-by-model bar chart (mean +/- std)
  * runs/comparison_curves.png      — progress curves of a few shared episodes,
                                       overlaid across models
  * a markdown table to stdout

Episodes are loaded deterministically by topreward_test.py, so episode_index
aligns across models and curves are directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(run_dir: Path) -> list[dict]:
    p = run_dir / "topreward.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open() if line.strip()]


def _stats(vals: list[float]) -> dict:
    a = np.array([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--curve-episodes", type=int, nargs="*", default=[4, 6, 22, 0, 19, 3])
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    # Discover model run dirs (those containing topreward.jsonl).
    model_dirs = sorted(d for d in runs.iterdir() if d.is_dir() and (d / "topreward.jsonl").exists())
    if not model_dirs:
        print(f"No model runs found under {runs}/*/topreward.jsonl")
        return

    per_model: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for d in model_dirs:
        rows = [r for r in _load(d) if r.get("error") is None]
        if not rows:
            continue
        tag = d.name
        per_model[tag] = rows
        summary[tag] = {
            "num_valid": len(rows),
            "voc": _stats([r["voc"] for r in rows]),
            "reward_mean": _stats([r["reward_mean"] for r in rows]),
            "answer_prob": _stats([r["answer_token_prob"] for r in rows]),
        }

    (runs / "comparison.json").write_text(json.dumps(summary, indent=2))

    # --- markdown table ---
    print(f"\n{'model':<32} {'n':>4} {'VOC mean':>9} {'VOC std':>8} {'reward':>8} {'P(True)':>9}")
    print("-" * 76)
    for tag, s in sorted(summary.items(), key=lambda kv: kv[1]["voc"]["mean"], reverse=True):
        v, r, p = s["voc"], s["reward_mean"], s["answer_prob"]
        print(f"{tag:<32} {s['num_valid']:>4} {v['mean']:>9.3f} {v['std']:>8.3f} {r['mean']:>8.2f} {p['mean']:>9.4f}")

    # --- VOC bar chart ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(summary, key=lambda t: summary[t]["voc"]["mean"], reverse=True)
    means = [summary[t]["voc"]["mean"] for t in order]
    stds = [summary[t]["voc"]["std"] for t in order]
    labels = [t.replace("Qwen_", "") for t in order]

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(order)), 5), constrained_layout=True)
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color="#4c72b0", alpha=0.9)
    ax.set_ylabel("VOC (Spearman: progress vs. chronological order)")
    ax.set_title(f"TOPReward progress-monotonicity (VOC) by model — {summary[order[0]]['num_valid']} Ego4D samples")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    for b, m in zip(bars, means, strict=False):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.3f}", ha="center", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.savefig(runs / "comparison_voc.png", dpi=120)
    plt.close(fig)

    # --- progress-curve overlays for a few shared episodes ---
    eps = args.curve_episodes
    ncol = min(3, len(eps))
    nrow = int(np.ceil(len(eps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), constrained_layout=True, squeeze=False)
    cmap = plt.get_cmap("tab10")
    model_color = {t: cmap(i % 10) for i, t in enumerate(order)}
    for k, ep in enumerate(eps):
        ax = axes[k // ncol][k % ncol]
        cap = ""
        for tag in order:
            row = next((r for r in per_model[tag] if r["episode_index"] == ep), None)
            if row is None or not row.get("progress"):
                continue
            cap = row["caption"]
            ax.plot(row["prefix_frame_counts"], row["progress"], "-o", ms=3, lw=1.5,
                    color=model_color[tag], label=tag.replace("Qwen_", ""))
        ax.set_title(f"ep {ep}: {cap[:48]}", fontsize=8, loc="left")
        ax.set_xlabel("prefix length (# frames)")
        ax.set_ylabel("progress")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7, loc="lower right")
    for k in range(len(eps), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Per-episode predicted progress curves across models", fontsize=11)
    fig.savefig(runs / "comparison_curves.png", dpi=120)
    plt.close(fig)

    print(f"\n[done] {runs}/comparison.json")
    print(f"[done] {runs}/comparison_voc.png")
    print(f"[done] {runs}/comparison_curves.png")


if __name__ == "__main__":
    main()
