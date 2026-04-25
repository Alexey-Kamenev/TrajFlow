# Copyright (c) 2025-present, Qi Yan.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Bird's-eye motion visualization for Waymo-style eval pickles."""

from __future__ import annotations

import io
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# Matplotlib is imported lazily inside plotting functions so importing this module
# stays light for training code paths that never render.

MapPolyline = Tuple[np.ndarray, float]  # (N, 2) xy, gray level 0-1


def _to_scalar_id(scenario_id: Any) -> str:
    x = np.asarray(scenario_id).reshape(-1)
    if x.size == 0:
        return ""
    v = x.flat[0]
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


def group_predictions_by_scene(pred_dicts: Sequence[Any]) -> List[List[dict]]:
    """Turn eval pickle contents into a list of per-scene agent predictions."""
    scenes: dict[str, List[dict]] = defaultdict(list)
    for item in pred_dicts:
        if isinstance(item, list):
            for d in item:
                if isinstance(d, dict) and "pred_trajs" in d:
                    sid = _to_scalar_id(d["scenario_id"])
                    scenes[sid].append(d)
        elif isinstance(item, dict) and "pred_trajs" in item:
            sid = _to_scalar_id(item["scenario_id"])
            scenes[sid].append(item)
    return list(scenes.values())


def load_scene_pickle(scenes_dir: str, scenario_id: str) -> dict:
    path = f"{scenes_dir.rstrip('/')}/sample_{scenario_id}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _collect_map_polylines(map_infos: dict) -> List[MapPolyline]:
    """Extract disjoint xy polylines from Waymo preprocess map_infos."""
    all_polylines = map_infos.get("all_polylines", np.zeros((0, 7), dtype=np.float32))
    segs: List[MapPolyline] = []
    for key in ("road_edge", "road_line", "lane", "crosswalk", "speed_bump", "driveway"):
        for item in map_infos.get(key, []) or []:
            s, e = item["polyline_index"]
            xy = np.asarray(all_polylines[s:e, :2], dtype=np.float64)
            if xy.shape[0] < 2:
                continue
            if key == "road_edge":
                tone = 0.35
            elif key == "road_line":
                tone = 0.5
            elif key == "lane":
                tone = 0.65
            else:
                tone = 0.55
            segs.append((xy, tone))
    return segs


def _rot2d(h: float) -> np.ndarray:
    c, s = np.cos(h), np.sin(h)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _agent_box_xy(center_xy: np.ndarray, heading: float, length: float, width: float) -> np.ndarray:
    """Corners of a centered box in world frame (xy only)."""
    R = _rot2d(heading)
    hl, hw = length / 2.0, width / 2.0
    corners = np.array(
        [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw]], dtype=np.float64
    )
    return (R @ corners.T).T + center_xy.reshape(1, 2)


def _annotate_bbox_agent_id(
    ax: Any,
    box: np.ndarray,
    object_id: int,
    *,
    fontsize: float = 7.0,
    zorder: int = 7,
) -> None:
    """Label the center of a closed bbox polyline with the track ``object_id``."""
    center = box[:-1].mean(axis=0)
    ax.text(
        float(center[0]),
        float(center[1]),
        str(int(object_id)),
        ha="center",
        va="center",
        fontsize=fontsize,
        color="0.08",
        zorder=zorder,
        clip_on=True,
    )


def _past_xy(traj: np.ndarray, mask: np.ndarray | None, upto: int) -> np.ndarray:
    """Trajectory xy up to index ``upto`` (inclusive)."""
    upto = min(upto, traj.shape[0] - 1)
    xy = traj[: upto + 1, :2].astype(np.float64)
    if mask is not None:
        m = mask[: upto + 1].astype(bool)
        xy = xy[m]
    else:
        valid = traj[: upto + 1, -1] > 0.5 if traj.shape[1] >= 10 else np.ones(len(xy), bool)
        xy = xy[valid]
    return xy


def _future_gt_xy(gt_full: np.ndarray, current_time_index: int) -> np.ndarray | None:
    if gt_full is None or gt_full.size == 0:
        return None
    fut = gt_full[current_time_index + 1 :, :2].astype(np.float64)
    if gt_full.shape[1] >= 10:
        valid = gt_full[current_time_index + 1 :, -1] > 0.5
        fut = fut[valid]
    if fut.shape[0] == 0:
        return None
    return fut


def _sort_modes(pred: dict, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(pred["pred_scores"], dtype=np.float64).reshape(-1)
    trajs = np.asarray(pred["pred_trajs"], dtype=np.float64)
    order = np.argsort(-scores)[:topk]
    return trajs[order], scores[order]


def _focal_indices(scene: dict, preds: Sequence[dict]) -> Tuple[np.ndarray, List[int]]:
    """Object indices in ``track_infos`` for each prediction dict (world-frame centers)."""
    obj_ids = np.asarray(scene["track_infos"]["object_id"])
    idxs: List[int] = []
    for pred in preds:
        oid = int(np.asarray(pred["object_id"]).reshape(-1)[0])
        idxs.append(int(np.where(obj_ids == oid)[0][0]))
    return obj_ids, idxs


def _bev_center_half_extent(
    scene: dict,
    preds: Sequence[dict],
    *,
    margin_m: float,
    topk_modes: int,
    pad_m: float = 5.0,
) -> Tuple[np.ndarray, float]:
    """Square BEV window covering all focal states, top-k preds, and GT futures."""
    current_time_index = int(scene["current_time_index"])
    trajs_full = np.asarray(scene["track_infos"]["trajs"], dtype=np.float64)
    chunks: List[np.ndarray] = []
    for pred in preds:
        _, idxs = _focal_indices(scene, [pred])
        fi = idxs[0]
        chunks.append(trajs_full[fi, current_time_index, :2].reshape(1, 2))
        pt, _ = _sort_modes(pred, topk_modes)
        chunks.append(pt[:, :, :2].reshape(-1, 2))
        if "gt_trajs" in pred:
            gt = _future_gt_xy(np.asarray(pred["gt_trajs"]), current_time_index)
            if gt is not None:
                chunks.append(gt)
    xy = np.concatenate(chunks, axis=0)
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    center = (lo + hi) / 2.0
    rad = float(np.max(np.abs(xy - center.reshape(1, 2)))) + pad_m
    half = max(margin_m, rad)
    return center, half


def _agent_mode_colors(agent_idx: int, num_modes: int) -> List[Tuple[float, float, float, float]]:
    """Distinct colors per agent; modes vary along tab10 with an offset."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    base = (agent_idx * 3) % 10
    return [cmap((base + mi) % 10) for mi in range(num_modes)]


def _subplots_bev_and_score_bars(
    figsize: Tuple[float, float],
    dpi: int,
) -> Tuple[Any, Any, Any]:
    """Main BEV axis on top and a bottom strip of vertical score bars."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig_w = figsize[0]
    fig_h = figsize[1] * 1.12
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[4.0, 0.88], hspace=0.12)
    ax_bev = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])
    return fig, ax_bev, ax_bar


def _draw_mode_scores_barv(
    ax_bar: Any,
    scores: np.ndarray,
    colors: Sequence[Any],
    xticklabels: Sequence[str] | None = None,
) -> None:
    """Vertical bar chart under the BEV: one bar per mode (x), height = score."""
    ax_bar.clear()
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = int(s.size)
    if n == 0:
        ax_bar.set_axis_off()
        return

    x = np.arange(n, dtype=np.float64)
    clist = [colors[min(i, len(colors) - 1)] for i in range(n)]
    bar_w = min(0.78, 4.8 / max(n, 1))
    ax_bar.bar(x, s, width=bar_w, color=clist, edgecolor="0.2", linewidth=0.25, align="center")
    ax_bar.set_xticks(x)
    if xticklabels is not None and len(xticklabels) == n:
        labels = list(xticklabels)
    else:
        labels = [str(i) for i in range(n)]
    rot = 40 if max((len(str(t)) for t in labels), default=0) > 3 else 0
    ax_bar.set_xticklabels(labels, fontsize=5.5, rotation=rot, ha="right" if rot else "center")

    smin_v, smax_v = float(s.min()), float(s.max())
    pad = max((smax_v - smin_v) * 0.08, 1e-4)
    if bool(np.all(s >= 0)):
        y_lo = 0.0
        y_hi = max(smax_v + pad, 1e-3)
    else:
        y_lo = smin_v - pad
        y_hi = smax_v + pad
    if y_lo >= y_hi:
        y_hi = y_lo + 1e-3
    ax_bar.set_ylim(y_lo, y_hi)
    ax_bar.set_xlim(-0.5, n - 0.5)
    ax_bar.set_ylabel("score", fontsize=7, labelpad=1)
    ax_bar.set_xlabel("mode", fontsize=6.5, labelpad=0.5)
    ax_bar.tick_params(axis="y", labelsize=6, length=2, pad=1)
    ax_bar.tick_params(axis="x", length=0, pad=0)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.4, linewidth=0.6)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)
    ax_bar.set_title("scores", fontsize=7.5, pad=3)
    ax_bar.margins(x=0.02, y=0.02)


def render_scene_multi_agent_gif(
    scene: dict,
    preds: Sequence[dict],
    out_path: str,
    *,
    topk_modes: int = 6,
    future_frame_stride: int = 4,
    figsize: Tuple[float, float] = (10.0, 10.0),
    dpi: int = 115,
    margin_m: float = 30.0,
    fps: float = 8.0,
) -> None:
    """One GIF: map, neighbors, and multiple agents' histories, boxes, and world-frame preds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    if not preds:
        raise ValueError("preds must be non-empty.")

    current_time_index = int(scene["current_time_index"])
    track_infos = scene["track_infos"]
    obj_types = np.asarray(track_infos["object_type"])
    trajs_full = np.asarray(track_infos["trajs"], dtype=np.float64)
    _, focal_idxs = _focal_indices(scene, preds)
    focal_set = set(focal_idxs)

    center_xy, half = _bev_center_half_extent(scene, preds, margin_m=margin_m, topk_modes=topk_modes)
    map_segs = _collect_map_polylines(scene["map_infos"])

    per_agent: List[Dict[str, Any]] = []
    max_steps = 0
    for pred, fidx in zip(preds, focal_idxs):
        pred_trajs, pred_scores = _sort_modes(pred, topk_modes)
        num_modes, num_steps, _ = pred_trajs.shape
        max_steps = max(max_steps, int(num_steps))
        gt_future = None
        if "gt_trajs" in pred:
            gt_future = _future_gt_xy(np.asarray(pred["gt_trajs"]), current_time_index)
        smin, smax = float(pred_scores.min()), float(pred_scores.max())
        score_rng = smax - smin + 1e-6
        heading = float(trajs_full[fidx, current_time_index, 6])
        per_agent.append(
            {
                "pred": pred,
                "fidx": fidx,
                "pred_trajs": pred_trajs,
                "pred_scores": pred_scores,
                "num_modes": num_modes,
                "num_steps": num_steps,
                "gt_future": gt_future,
                "smin": smin,
                "score_rng": score_rng,
                "heading": heading,
                "hist": _past_xy(trajs_full[fidx], None, current_time_index),
                "mode_colors": _agent_mode_colors(len(per_agent), num_modes),
            }
        )

    neighbor_lines: List[Tuple[np.ndarray, str]] = []
    for oi in range(trajs_full.shape[0]):
        if oi in focal_set:
            continue
        past = _past_xy(trajs_full[oi], None, current_time_index)
        if past.shape[0] >= 2:
            neighbor_lines.append((past, str(obj_types[oi])))

    frame_indices: List[int] = []
    j = 0
    while j < max_steps:
        frame_indices.append(j)
        j += future_frame_stride
    if frame_indices[-1] != max_steps - 1:
        frame_indices.append(max_steps - 1)

    def draw_frame(ax: Any, fi: int) -> None:
        ax.clear()
        ax.set_aspect("equal")
        ax.set_axis_off()

        for xy, tone in map_segs:
            ax.plot(xy[:, 0], xy[:, 1], color=(tone, tone, tone), linewidth=0.8, solid_capstyle="round")

        for past, ot in neighbor_lines:
            col = (0.55, 0.55, 0.75, 0.45) if "PED" in ot else (0.45, 0.55, 0.65, 0.5)
            ax.plot(past[:, 0], past[:, 1], color=col, linewidth=1.2)

        for oi in range(trajs_full.shape[0]):
            if oi in focal_set:
                continue
            st = trajs_full[oi, current_time_index]
            if st.shape[0] >= 10 and st[-1] < 0.5:
                continue
            ax.scatter([st[0]], [st[1]], s=10, c=[(0.4, 0.45, 0.55)])

        for pa in per_agent:
            fidx = pa["fidx"]
            pred_trajs = pa["pred_trajs"]
            pred_scores = pa["pred_scores"]
            hist = pa["hist"]
            if hist.shape[0] >= 2:
                ax.plot(hist[:, 0], hist[:, 1], color="0.15", linewidth=2.0, zorder=4)
            L = float(trajs_full[fidx, current_time_index, 3])
            W = float(trajs_full[fidx, current_time_index, 4])
            box = _agent_box_xy(
                trajs_full[fidx, current_time_index, :2],
                pa["heading"],
                max(L, 2.0),
                max(W, 1.0),
            )
            ax.fill(box[:, 0], box[:, 1], facecolor="0.88", edgecolor="0.2", linewidth=1.0, zorder=5)
            oid = int(np.asarray(pa["pred"]["object_id"]).reshape(-1)[0])
            _annotate_bbox_agent_id(ax, box, oid, zorder=8)

            nsteps = pa["num_steps"]
            eff_fi = min(fi, nsteps - 1)
            for mi in range(pa["num_modes"]):
                seg = pred_trajs[mi, : eff_fi + 1, :2]
                if seg.shape[0] >= 2:
                    ax.plot(
                        seg[:, 0],
                        seg[:, 1],
                        color=pa["mode_colors"][mi],
                        linewidth=1.7,
                        alpha=0.5
                        + 0.4 * float(pred_scores[mi] - pa["smin"]) / pa["score_rng"],
                        zorder=3,
                    )
                ax.scatter(
                    [seg[-1, 0]],
                    [seg[-1, 1]],
                    s=26,
                    color=[pa["mode_colors"][mi]],
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=6,
                )

            gt_future = pa["gt_future"]
            if gt_future is not None:
                g_len = min(eff_fi + 1, len(gt_future))
                gseg = gt_future[:g_len]
                if gseg.shape[0] >= 2:
                    ax.plot(gseg[:, 0], gseg[:, 1], color="#1a7f37", linewidth=2.0, linestyle="--", zorder=4)

        ax.set_xlim(center_xy[0] - half, center_xy[0] + half)
        ax.set_ylim(center_xy[1] - half, center_xy[1] + half)

        oids = [int(np.asarray(p["object_id"]).reshape(-1)[0]) for p in preds]
        sid = _to_scalar_id(preds[0]["scenario_id"])
        ax.set_title(f"Multi-agent  |  scenario {sid}  |  ids {oids}", fontsize=10)

    scores_flat: List[float] = []
    colors_flat: List[Any] = []
    ylabels_flat: List[str] = []
    for ai, pa in enumerate(per_agent):
        oid = int(np.asarray(pa["pred"]["object_id"]).reshape(-1)[0])
        oid_short = str(oid)[-5:] if len(str(oid)) > 5 else str(oid)
        for mi in range(pa["num_modes"]):
            scores_flat.append(float(pa["pred_scores"][mi]))
            colors_flat.append(pa["mode_colors"][mi])
            ylabels_flat.append(f"{oid_short}:{mi}")

    fig, ax_bev, ax_bar = _subplots_bev_and_score_bars(figsize, dpi)
    _draw_mode_scores_barv(ax_bar, np.asarray(scores_flat, dtype=np.float64), colors_flat, ylabels_flat)
    frames_pil: List[Any] = []
    for fi in frame_indices:
        draw_frame(ax_bev, fi)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.08)
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
        frames_pil.append(im.copy())
        im.close()
        buf.close()

    duration_ms = int(1000.0 / max(fps, 0.1))
    frames_pil[0].save(
        out_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    for im in frames_pil:
        im.close()
    plt.close(fig)


def render_scene_multi_agent_png(
    scene: dict,
    preds: Sequence[dict],
    out_path: str,
    *,
    topk_modes: int = 6,
    show_future_index: int | None = None,
    figsize: Tuple[float, float] = (10.0, 10.0),
    dpi: int = 135,
    margin_m: float = 30.0,
) -> None:
    """Single PNG with multiple agents' world-frame predictions on one BEV."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not preds:
        raise ValueError("preds must be non-empty.")

    current_time_index = int(scene["current_time_index"])
    track_infos = scene["track_infos"]
    trajs_full = np.asarray(track_infos["trajs"], dtype=np.float64)
    obj_types = np.asarray(track_infos["object_type"])
    _, focal_idxs = _focal_indices(scene, preds)
    focal_set = set(focal_idxs)

    per_agent: List[Dict[str, Any]] = []
    max_steps = 0
    for pred, fidx in zip(preds, focal_idxs):
        pred_trajs, pred_scores = _sort_modes(pred, topk_modes)
        num_modes, num_steps, _ = pred_trajs.shape
        max_steps = max(max_steps, int(num_steps))
        per_agent.append(
            {
                "fidx": fidx,
                "pred_trajs": pred_trajs,
                "pred_scores": pred_scores,
                "num_modes": num_modes,
                "num_steps": num_steps,
                "smin": float(pred_scores.min()),
                "score_rng": float(pred_scores.max() - pred_scores.min()) + 1e-6,
                "heading": float(trajs_full[fidx, current_time_index, 6]),
                "hist": _past_xy(trajs_full[fidx], None, current_time_index),
                "gt_future": _future_gt_xy(np.asarray(pred["gt_trajs"]), current_time_index)
                if "gt_trajs" in pred
                else None,
                "mode_colors": _agent_mode_colors(len(per_agent), num_modes),
            }
        )

    fi_default = max_steps - 1
    fi = fi_default if show_future_index is None else int(np.clip(show_future_index, 0, max_steps - 1))

    center_xy, half = _bev_center_half_extent(scene, preds, margin_m=margin_m, topk_modes=topk_modes)
    map_segs = _collect_map_polylines(scene["map_infos"])

    scores_flat: List[float] = []
    colors_flat: List[Any] = []
    ylabels_flat: List[str] = []
    for ai, pa in enumerate(per_agent):
        oid = int(np.asarray(preds[ai]["object_id"]).reshape(-1)[0])
        oid_short = str(oid)[-5:] if len(str(oid)) > 5 else str(oid)
        for mi in range(pa["num_modes"]):
            scores_flat.append(float(pa["pred_scores"][mi]))
            colors_flat.append(pa["mode_colors"][mi])
            ylabels_flat.append(f"{oid_short}:{mi}")

    fig, ax, ax_bar = _subplots_bev_and_score_bars(figsize, dpi)
    _draw_mode_scores_barv(ax_bar, np.asarray(scores_flat, dtype=np.float64), colors_flat, ylabels_flat)

    for xy, tone in map_segs:
        ax.plot(xy[:, 0], xy[:, 1], color=(tone, tone, tone), linewidth=0.8, solid_capstyle="round")

    for oi in range(trajs_full.shape[0]):
        if oi in focal_set:
            continue
        past = _past_xy(trajs_full[oi], None, current_time_index)
        if past.shape[0] >= 2:
            ot = str(obj_types[oi])
            col = (0.55, 0.55, 0.75, 0.45) if "PED" in ot else (0.45, 0.55, 0.65, 0.5)
            ax.plot(past[:, 0], past[:, 1], color=col, linewidth=1.0)

    for ai, pa in enumerate(per_agent):
        fidx = pa["fidx"]
        eff_fi = min(fi, pa["num_steps"] - 1)
        hist = pa["hist"]
        if hist.shape[0] >= 2:
            ax.plot(hist[:, 0], hist[:, 1], color="0.15", linewidth=2.0)
        L = float(trajs_full[fidx, current_time_index, 3])
        W = float(trajs_full[fidx, current_time_index, 4])
        box = _agent_box_xy(trajs_full[fidx, current_time_index, :2], pa["heading"], max(L, 2.0), max(W, 1.0))
        ax.fill(box[:, 0], box[:, 1], facecolor="0.88", edgecolor="0.2", linewidth=1.0)
        oid = int(np.asarray(preds[ai]["object_id"]).reshape(-1)[0])
        _annotate_bbox_agent_id(ax, box, oid)

        pred_trajs = pa["pred_trajs"]
        pred_scores = pa["pred_scores"]
        for mi in range(pa["num_modes"]):
            seg = pred_trajs[mi, : eff_fi + 1, :2]
            if seg.shape[0] >= 2:
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    color=pa["mode_colors"][mi],
                    linewidth=1.7,
                    alpha=0.5 + 0.4 * float(pred_scores[mi] - pa["smin"]) / pa["score_rng"],
                )

        gt_future = pa["gt_future"]
        if gt_future is not None:
            gfi = min(eff_fi, len(gt_future) - 1)
            gseg = gt_future[: gfi + 1]
            if gseg.shape[0] >= 2:
                ax.plot(gseg[:, 0], gseg[:, 1], color="#1a7f37", linewidth=2.0, linestyle="--")

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(center_xy[0] - half, center_xy[0] + half)
    ax.set_ylim(center_xy[1] - half, center_xy[1] + half)
    oids = [int(np.asarray(p["object_id"]).reshape(-1)[0]) for p in preds]
    sid = _to_scalar_id(preds[0]["scenario_id"])
    ax.set_title(f"Multi-agent  |  {sid}  |  ids {oids}", fontsize=10)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_scene_agent_gif(
    scene: dict,
    pred: dict,
    out_path: str,
    *,
    topk_modes: int = 6,
    future_frame_stride: int = 4,
    figsize: Tuple[float, float] = (10.0, 10.0),
    dpi: int = 115,
    margin_m: float = 30.0,
    fps: float = 8.0,
) -> None:
    """Write one animated GIF: map, neighbors, history, unfolding multi-modal futures + GT."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    current_time_index = int(scene["current_time_index"])
    track_infos = scene["track_infos"]
    obj_ids = np.asarray(track_infos["object_id"])
    obj_types = np.asarray(track_infos["object_type"])
    trajs_full = np.asarray(track_infos["trajs"], dtype=np.float64)

    focal_oid = int(np.asarray(pred["object_id"]).reshape(-1)[0])
    focal_idx = int(np.where(obj_ids == focal_oid)[0][0])

    center_xy = trajs_full[focal_idx, current_time_index, :2]
    heading = float(trajs_full[focal_idx, current_time_index, 6])

    map_segs = _collect_map_polylines(scene["map_infos"])

    pred_trajs, pred_scores = _sort_modes(pred, topk_modes)
    num_modes, num_steps, _ = pred_trajs.shape
    gt_future = None
    if "gt_trajs" in pred:
        gt_future = _future_gt_xy(np.asarray(pred["gt_trajs"]), current_time_index)

    cmap = plt.get_cmap("tab10")
    mode_colors = [cmap(i % 10) for i in range(num_modes)]
    smin, smax = float(pred_scores.min()), float(pred_scores.max())
    score_rng = smax - smin + 1e-6

    # Build neighbor past polylines once (static background).
    neighbor_lines: List[Tuple[np.ndarray, str]] = []
    for oi in range(trajs_full.shape[0]):
        if oi == focal_idx:
            continue
        past = _past_xy(trajs_full[oi], None, current_time_index)
        if past.shape[0] >= 2:
            neighbor_lines.append((past, str(obj_types[oi])))

    focal_hist = _past_xy(trajs_full[focal_idx], None, current_time_index)

    # Animation covers cumulative future indices 0..num_steps-1 (strided).
    frame_indices: List[int] = []
    j = 0
    while j < num_steps:
        frame_indices.append(j)
        j += future_frame_stride
    if frame_indices[-1] != num_steps - 1:
        frame_indices.append(num_steps - 1)

    def draw_frame(ax: Any, fi: int) -> None:
        ax.clear()
        ax.set_aspect("equal")
        ax.set_axis_off()

        # Map.
        for xy, tone in map_segs:
            ax.plot(xy[:, 0], xy[:, 1], color=(tone, tone, tone), linewidth=0.8, solid_capstyle="round")

        # Neighbors (light).
        for past, ot in neighbor_lines:
            col = (0.55, 0.55, 0.75, 0.45) if "PED" in ot else (0.45, 0.55, 0.65, 0.5)
            ax.plot(past[:, 0], past[:, 1], color=col, linewidth=1.2)

        # Neighbor current positions (dots).
        for oi in range(trajs_full.shape[0]):
            if oi == focal_idx:
                continue
            st = trajs_full[oi, current_time_index]
            if st.shape[0] >= 10 and st[-1] < 0.5:
                continue
            ax.scatter([st[0]], [st[1]], s=10, c=[(0.4, 0.45, 0.55)])

        # Focal history.
        if focal_hist.shape[0] >= 2:
            ax.plot(focal_hist[:, 0], focal_hist[:, 1], color="0.1", linewidth=2.5, zorder=4)

        L, W = float(trajs_full[focal_idx, current_time_index, 3]), float(trajs_full[focal_idx, current_time_index, 4])
        box = _agent_box_xy(trajs_full[focal_idx, current_time_index, :2], heading, max(L, 2.0), max(W, 1.0))
        ax.fill(box[:, 0], box[:, 1], facecolor="0.85", edgecolor="0.15", linewidth=1.2, zorder=5)
        _annotate_bbox_agent_id(ax, box, focal_oid, zorder=8)

        # Predictions (cumulative up to fi).
        for mi in range(num_modes):
            seg = pred_trajs[mi, : fi + 1, :2]
            if seg.shape[0] >= 2:
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    color=mode_colors[mi],
                    linewidth=1.8,
                    alpha=0.55 + 0.35 * float(pred_scores[mi] - smin) / score_rng,
                    zorder=3,
                )
            ax.scatter(
                [seg[-1, 0]],
                [seg[-1, 1]],
                s=28,
                color=[mode_colors[mi]],
                edgecolors="white",
                linewidths=0.4,
                zorder=6,
            )

        if gt_future is not None:
            g_len = min(fi + 1, len(gt_future))
            gseg = gt_future[:g_len]
            if gseg.shape[0] >= 2:
                ax.plot(gseg[:, 0], gseg[:, 1], color="#1a7f37", linewidth=2.4, linestyle="--", zorder=4)

        ax.set_xlim(center_xy[0] - margin_m, center_xy[0] + margin_m)
        ax.set_ylim(center_xy[1] - margin_m, center_xy[1] + margin_m)

        ot = str(pred.get("object_type", ""))
        ax.set_title(
            f"{ot.replace('TYPE_', '').title()}  |  scenario {_to_scalar_id(pred['scenario_id'])}", fontsize=10
        )

    fig, ax, ax_bar = _subplots_bev_and_score_bars(figsize, dpi)
    _draw_mode_scores_barv(ax_bar, pred_scores, mode_colors)
    frames_pil: List[Any] = []
    for fi in frame_indices:
        draw_frame(ax, fi)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.08)
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
        frames_pil.append(im.copy())
        im.close()
        buf.close()

    duration_ms = int(1000.0 / max(fps, 0.1))
    frames_pil[0].save(
        out_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    for im in frames_pil:
        im.close()
    plt.close(fig)


def render_scene_agent_png(
    scene: dict,
    pred: dict,
    out_path: str,
    *,
    topk_modes: int = 6,
    show_future_index: int | None = None,
    figsize: Tuple[float, float] = (10.0, 10.0),
    dpi: int = 135,
    margin_m: float = 30.0,
) -> None:
    """Single-frame PNG at a chosen future index (default: full horizon)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_trajs, pred_scores = _sort_modes(pred, topk_modes)
    num_steps = pred_trajs.shape[1]
    fi = num_steps - 1 if show_future_index is None else int(np.clip(show_future_index, 0, num_steps - 1))
    smin, smax = float(pred_scores.min()), float(pred_scores.max())
    score_rng = smax - smin + 1e-6

    # Single-frame snapshot (same geometry as the GIF).
    current_time_index = int(scene["current_time_index"])
    track_infos = scene["track_infos"]
    obj_ids = np.asarray(track_infos["object_id"])
    trajs_full = np.asarray(track_infos["trajs"], dtype=np.float64)
    focal_oid = int(np.asarray(pred["object_id"]).reshape(-1)[0])
    focal_idx = int(np.where(obj_ids == focal_oid)[0][0])
    center_xy = trajs_full[focal_idx, current_time_index, :2]
    heading = float(trajs_full[focal_idx, current_time_index, 6])
    map_segs = _collect_map_polylines(scene["map_infos"])
    cmap = plt.get_cmap("tab10")
    num_modes = pred_trajs.shape[0]
    mode_colors = [cmap(i % 10) for i in range(num_modes)]

    gt_future = None
    if "gt_trajs" in pred:
        gt_future = _future_gt_xy(np.asarray(pred["gt_trajs"]), current_time_index)

    fig, ax, ax_bar = _subplots_bev_and_score_bars(figsize, dpi)
    _draw_mode_scores_barv(ax_bar, pred_scores, mode_colors)

    for xy, tone in map_segs:
        ax.plot(xy[:, 0], xy[:, 1], color=(tone, tone, tone), linewidth=0.8, solid_capstyle="round")

    for oi in range(trajs_full.shape[0]):
        if oi == focal_idx:
            continue
        past = _past_xy(trajs_full[oi], None, current_time_index)
        if past.shape[0] >= 2:
            ax.plot(past[:, 0], past[:, 1], color=(0.55, 0.55, 0.75, 0.45), linewidth=1.0)

    focal_hist = _past_xy(trajs_full[focal_idx], None, current_time_index)
    if focal_hist.shape[0] >= 2:
        ax.plot(focal_hist[:, 0], focal_hist[:, 1], color="0.1", linewidth=2.5)

    L, W = float(trajs_full[focal_idx, current_time_index, 3]), float(trajs_full[focal_idx, current_time_index, 4])
    box = _agent_box_xy(trajs_full[focal_idx, current_time_index, :2], heading, max(L, 2.0), max(W, 1.0))
    ax.fill(box[:, 0], box[:, 1], facecolor="0.85", edgecolor="0.15", linewidth=1.2)
    _annotate_bbox_agent_id(ax, box, focal_oid)

    for mi in range(num_modes):
        seg = pred_trajs[mi, : fi + 1, :2]
        if seg.shape[0] >= 2:
            ax.plot(
                seg[:, 0],
                seg[:, 1],
                color=mode_colors[mi],
                linewidth=1.8,
                alpha=0.55 + 0.35 * float(pred_scores[mi] - smin) / score_rng,
            )

    if gt_future is not None:
        gfi = min(fi, len(gt_future) - 1)
        gseg = gt_future[: gfi + 1]
        if gseg.shape[0] >= 2:
            ax.plot(gseg[:, 0], gseg[:, 1], color="#1a7f37", linewidth=2.4, linestyle="--")

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(center_xy[0] - margin_m, center_xy[0] + margin_m)
    ax.set_ylim(center_xy[1] - margin_m, center_xy[1] + margin_m)
    ot = str(pred.get("object_type", ""))
    ax.set_title(f"{ot.replace('TYPE_', '').title()}  |  {_to_scalar_id(pred['scenario_id'])}", fontsize=10)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
