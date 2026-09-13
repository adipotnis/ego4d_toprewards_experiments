"""Compare whole-episode runs under runs/<tag>/topreward.jsonl.

Write aggregate JSON, VOC bars, and shared-episode curves; print a summary
table. Episode indices align because the loader uses dataset order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import topreward_test as tr


def _short_tag(tag: str) -> str:
    return tag.replace("Qwen_", "").replace("allenai_", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--curve-episodes", type=int, nargs="*", default=[4, 6, 22, 0, 19, 3])
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    model_dirs = sorted(path.parent for path in runs.glob("*/topreward.jsonl"))
    if not model_dirs:
        print(f"No model runs found under {runs}/*/topreward.jsonl")
        return

    by_ep: dict[str, dict[int, dict]] = {}  # tag -> episode_index -> row
    summary: dict[str, dict] = {}
    for d in model_dirs:
        rows = [r for r in tr.read_jsonl(d / "topreward.jsonl") if r.get("error") is None and "reward_mean" in r]
        if not rows:
            continue
        tag = d.name
        by_ep[tag] = {r["episode_index"]: r for r in rows}
        summary[tag] = {
            "num_valid": len(rows),
            "voc": tr.summary_stats(r["voc"] for r in rows),
            "reward_mean": tr.summary_stats(r["reward_mean"] for r in rows),
            "answer_prob": tr.summary_stats(r["answer_token_prob"] for r in rows),
        }

    if not summary:
        print(f"No valid whole-episode scores found under {runs}")
        return

    (runs / "comparison.json").write_text(json.dumps(summary, indent=2))

    # Models ordered by mean VOC, best first; used by the table and both figures.
    order = sorted(summary, key=lambda t: summary[t]["voc"]["mean"], reverse=True)

    print(f"\n{'model':<32} {'n':>4} {'VOC mean':>9} {'VOC std':>8} {'reward':>8} {'P(True)':>9}")
    print("-" * 76)
    for tag in order:
        s = summary[tag]
        v, r, p = s["voc"], s["reward_mean"], s["answer_prob"]
        print(f"{tag:<32} {s['num_valid']:>4} {v['mean']:>9.3f} {v['std']:>8.3f} {r['mean']:>8.2f} {p['mean']:>9.4f}")

    plt = tr.agg_pyplot()

    means = [summary[t]["voc"]["mean"] for t in order]
    stds = [summary[t]["voc"]["std"] for t in order]
    labels = [_short_tag(t) for t in order]

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

    eps = args.curve_episodes
    if not eps:
        return
    ncol = min(3, len(eps))
    nrow = int(np.ceil(len(eps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), constrained_layout=True, squeeze=False)
    cmap = plt.get_cmap("tab10")
    model_color = {t: cmap(i % 10) for i, t in enumerate(order)}
    for k, ep in enumerate(eps):
        ax = axes[k // ncol][k % ncol]
        cap = ""
        for tag in order:
            row = by_ep[tag].get(ep)
            if row is None or not row.get("progress"):
                continue
            cap = row["caption"]
            ax.plot(row["prefix_frame_counts"], row["progress"], "-o", ms=3, lw=1.5, color=model_color[tag], label=_short_tag(tag))
        ax.set_title(f"ep {ep}: {cap[:48]}", fontsize=8, loc="left")
        tr.style_progress_axis(ax)
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
