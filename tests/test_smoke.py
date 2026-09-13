"""Offline integration coverage for the experiment and its output consumers."""

import json
import subprocess
import sys
from types import SimpleNamespace

import av
import numpy as np
import torch

from ego4d_toprewards import core as tr
from ego4d_toprewards.cli import videos


def test_basic_run(tmp_path, monkeypatch):
    frames = [np.full((24, 32, 3), value, dtype=np.uint8) for value in (0, 40, 80, 120)]
    sample = tr.Ego4DSample(0, 1, "picks cup., pours water.", frames, 10)

    class TinyModel:
        def __call__(self, input_ids, strength):
            logits = torch.zeros(1, 4, 3)
            logits[:, :, 1] = strength
            return SimpleNamespace(logits=logits)

    def inputs(vlm, images, fps, prompt, scored):
        return {"input_ids": torch.tensor([[0, 0, 1, 1]]), "strength": len(images) / 4}, 2

    vlm = tr.VLM(TinyModel(), None, inputs)
    original_loader = tr.load_ego4d_samples
    monkeypatch.setattr(tr, "load_model", lambda name: vlm)
    monkeypatch.setattr(tr, "load_ego4d_samples", lambda *a, **kw: [sample])
    full = tr.logits_reward(vlm, frames, sample.caption, sample.fps)
    expected = torch.log_softmax(torch.tensor([0.0, 1.0, 0.0]), dim=0)[1].item()
    assert abs(full["reward_mean"] - expected) < 1e-6
    assert full["token_count"] == 2
    single = tr.score_clip(vlm, frames, sample.caption, sample.fps, num_prefixes=1)
    assert single["prefix_frame_counts"] == [4]
    assert single["reward_mean"] == full["reward_mean"]

    for split in (False, True):
        out = tmp_path / ("subgoals" if split else "model") / "topreward.jsonl"
        plots = out.parent / "plots"
        tr.run(1, "tiny", str(out), None, 4, 3, str(plots), split_subgoals=split)
        row = tr.read_jsonl(out)[0]
        scores = row["subgoals"] if split else [row]
        assert len(scores) == (2 if split else 1)
        assert all(s["error"] is None and s["progress"] == [0.0, s["progress"][1], 1.0] for s in scores)
        assert json.loads(out.with_suffix(".summary.json").read_text())["num_valid"] == len(scores)
        assert len(list(plots.glob("*.png"))) == (3 if split else 1)
        if split:
            mp4 = out.parent / "video.mp4"
            assert videos.render_episode_video(row, frames, mp4, 2, 2, dpi=30)
            with av.open(str(mp4)) as container:
                assert len(list(container.decode(video=0))) == 2

    subprocess.run([sys.executable, "-m", "ego4d_toprewards.cli.compare", "--runs-dir", str(tmp_path), "--curve-episodes", "0"], check=True)
    assert set(json.loads((tmp_path / "comparison.json").read_text())) == {"model"}
    assert (tmp_path / "comparison_curves.png").is_file()
    assert tr.split_subgoals(" picks cup., , pours water. ") == ["picks cup.", "pours water."]
    assert tr.summary_stats([None, float("nan"), float("inf"), 2])["mean"] == 2

    # Files with the same index in different chunks must decode distinct videos.
    import huggingface_hub
    import pyarrow.parquet as pq

    rows = [{"episode_index": i, "video_file_index": 0, "video_chunk_index": i, "start_frame": 0, "end_frame": 3, "task_index": 1} for i in range(2)]
    decoded = []
    with monkeypatch.context() as patches:
        patches.setattr(tr, "load_ego4d_samples", original_loader)
        patches.setattr(huggingface_hub, "hf_hub_download", lambda **kw: kw["filename"])
        patches.setattr(pq, "read_table", lambda path: SimpleNamespace(to_pydict=lambda: {"task_index": [1], "task": ["picks cup."]}))
        patches.setattr(tr, "_iter_episode_rows", lambda dl: iter(rows))
        patches.setattr(tr, "_decode_frames", lambda path, start, end: decoded.append(path) or frames)
        assert len(tr.load_ego4d_samples(2, None)) == 2
        assert len(set(decoded)) == 2
        assert tr.load_ego4d_samples(0, None) == []
