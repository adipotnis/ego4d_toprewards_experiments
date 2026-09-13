"""make_subgoal_videos.py — synced per-episode video for the subgoal run.

For each episode produced by `topreward_test.py --split-subgoals`, render an mp4:

    +-------------------------------------------------------------+
    |                  the Ego4D clip (plays smooth)              |
    +------------+------------+------------+ ... +----------------+
    | subgoal 0  | subgoal 1  | subgoal 2  |     |  subgoal N-1   |  <- side by side
    | progress   | progress   | progress   |     |  progress      |     (scaled down)
    +------------+------------+------------+ ... +----------------+

The real episode frames play in the top panel. A red cursor sweeps each
subgoal's progress curve left->right in sync with the video's time fraction,
with a dot riding the (interpolated) curve. Each subplot is titled with its
subgoal text + VOC.

Reads the curves from the run's JSONL (one EpisodeResult per line) and re-decodes
the full frames of exactly those episodes via `topreward_test.load_ego4d_samples`
(CPU only, uses the cached videos in HF_HOME). Encodes mp4 with PyAV (no ffmpeg
binary needed).

    python make_subgoal_videos.py \
        --jsonl runs/Qwen_Qwen3-VL-8B-Instruct_subgoals/topreward.jsonl \
        --out-dir runs/Qwen_Qwen3-VL-8B-Instruct_subgoals/videos
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import topreward_test as tr


def _even(n: int) -> int:
    return n - (n % 2)


def render_episode_video(ep: dict, frames: list, out_path: Path, max_render_frames: int, out_fps: int, dpi: int = 100) -> bool:
    """Render one episode's synced video. Returns True on success."""
    import av

    plt = tr.agg_pyplot()

    subs = [s for s in ep["subgoals"] if s.get("error") is None and s.get("prefix_frame_counts") and s.get("progress")]
    n = len(subs)
    if n == 0 or len(frames) < 2:
        return False

    # Subsample the top-panel frames for rendering (keeps encode time bounded while
    # still smooth). These are only for display; the curves are unchanged.
    vid = tr.uniform_subsample(frames, max_render_frames)
    total = len(vid)

    # --- figure: top video spans all columns, one subplot per subgoal below ---
    fig = plt.figure(figsize=(max(12.0, n * 1.55), 7.2), dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(2, n, height_ratios=[3.2, 2.0])

    axv = fig.add_subplot(gs[0, :])
    axv.axis("off")
    im = axv.imshow(vid[0])
    axv.set_title(
        f"ep {ep['episode_index']} — {n} subgoals scored over the full clip",
        loc="left",
        fontsize=11,
    )

    cursors = []  # (xs, ys, vline, dot)
    for j, s in enumerate(subs):
        ax = fig.add_subplot(gs[1, j])
        xs = np.asarray(s["prefix_frame_counts"], dtype=float)  # ascending
        ys = np.asarray(s["progress"], dtype=float)
        x0, x1 = float(xs[0]), float(xs[-1])
        ax.plot(xs, ys, ".-", color="#1f77b4", lw=1.4, ms=3)
        vline = ax.axvline(x0, color="crimson", lw=1.3)
        (dot,) = ax.plot([x0], [ys[0]], "o", color="crimson", ms=6, zorder=5)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlim(x0, x1)
        voc = s.get("voc", float("nan"))
        ax.set_title(f"[{s['subgoal_index']}] VOC={voc:.2f}\n{tr.wrap_title(s['caption'], width=22)}", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_xticks([x0, x1])
        if j == 0:
            ax.set_ylabel("progress [0,1]", fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("frames", fontsize=6)
        cursors.append((xs, ys, vline, dot))

    # Lock the layout once (constrained_layout re-solving every frame is slow and
    # can jitter the canvas size); after the first draw the geometry is stable.
    fig.canvas.draw()
    fig.set_layout_engine("none")

    w, h = fig.canvas.get_width_height()
    w, h = _even(w), _even(h)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=out_fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    stream.options = {"crf": "20"}

    for t in range(total):
        p = t / (total - 1) if total > 1 else 0.0
        im.set_data(vid[t])
        for xs, ys, vline, dot in cursors:
            cx = float(xs[0] + p * (xs[-1] - xs[0]))
            cy = float(np.interp(cx, xs, ys))
            vline.set_xdata([cx, cx])
            dot.set_data([cx], [cy])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:h, :w, :3]
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(buf), format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)

    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Render synced subgoal videos for a --split-subgoals run")
    p.add_argument("--jsonl", default="runs/Qwen_Qwen3-VL-8B-Instruct_subgoals/topreward.jsonl")
    p.add_argument("--out-dir", default="runs/Qwen_Qwen3-VL-8B-Instruct_subgoals/videos")
    p.add_argument("--cache-dir", default=None, help="HF cache dir (defaults to HF_HOME)")
    p.add_argument("--max-frames", type=int, default=60, help="max top-panel frames rendered per clip")
    p.add_argument("--fps", type=int, default=12, help="output video fps")
    p.add_argument("--episodes", default="", help="comma-separated episode indices to render (default: all in the jsonl)")
    args = p.parse_args()

    episodes = tr.read_jsonl(Path(args.jsonl))
    if args.episodes:
        want = {int(x) for x in args.episodes.split(",") if x.strip()}
        episodes = [e for e in episodes if e["episode_index"] in want]
    n_eps = len(episodes)

    # Re-decode full frames for exactly the episodes we need.
    need = {e["episode_index"] for e in episodes}
    samples = tr.load_ego4d_samples(len(need), cache_dir=args.cache_dir, episode_indices=need)
    frames_by_ep = {s.episode_index: s.frames for s in samples}

    out_dir = Path(args.out_dir)
    made = []
    for i, ep in enumerate(episodes):
        epi = ep["episode_index"]
        frames = frames_by_ep.get(epi)
        if frames is None:
            print(f"[vid] {i + 1}/{n_eps} ep={epi} SKIP (no frames)")
            continue
        out_path = out_dir / f"ep{epi:04d}.mp4"
        print(f"[vid] {i + 1}/{n_eps} ep={epi} subgoals={ep['num_subgoals']} frames={len(frames)} -> {out_path}")
        ok = render_episode_video(ep, frames, out_path, max_render_frames=args.max_frames, out_fps=args.fps)
        if ok:
            made.append(str(out_path))

    print(f"\n[done] rendered {len(made)} videos into {out_dir}")
    for m in made:
        print("   ", m)


if __name__ == "__main__":
    main()
