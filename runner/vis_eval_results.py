# Copyright (c) 2025-present, Qi Yan.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Render BEV GIFs/PNGs from eval output (result_denoiser.pkl) and Waymo processed scenes.

Example:

  python runner/vis_eval_results.py \\
    --result_pkl output/.../eval/epoch_10/result_denoiser.pkl \\
    --scenes_dir data/waymo/processed_scenarios_validation \\
    --out_dir output/.../eval/epoch_10/vis \\
    --max_scenes 5 --max_agents_per_scene 2 \\
    --combined_scene_output also
"""

import argparse
import os
import pickle
import sys
from typing import Tuple

import numpy as np

import _init_path  # noqa: F401  # isort: skip

from trajflow.utils import waymo_motion_vis


def _parse_figsize(s: str) -> Tuple[float, float]:
    """Parse ``W,H`` or ``WxH`` into figure size in inches."""
    raw = s.replace("x", ",").replace(" ", "")
    parts = [p for p in raw.split(",") if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("figsize must be two numbers, e.g. 10,10 or 12x9")
    return (float(parts[0]), float(parts[1]))


def parse_args():
    p = argparse.ArgumentParser(description="Visualize Waymo motion predictions from eval pickle.")
    p.add_argument("--result_pkl", type=str, required=True, help="Path to result_denoiser.pkl from eval.")
    p.add_argument(
        "--scenes_dir",
        type=str,
        required=True,
        help="Directory of sample_<scenario_id>.pkl (same as DATA_ROOT/SPLIT_DIR in yaml, e.g. .../processed_scenarios_validation).",
    )
    p.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <pkl_dir>/motion_vis).")
    p.add_argument("--max_scenes", type=int, default=8, help="Cap number of scenarios.")
    p.add_argument(
        "--max_agents_per_scene",
        type=int,
        default=3,
        help=(
            "Upper bound on how many prediction entries to render per scenario. "
            "Eval pickles usually have one entry per Waymo tracks_to_predict agent "
            "(validation/test commonly has 2 per scene), so raising this above that "
            "count does not add outputs until eval saves more agents per scene."
        ),
    )
    p.add_argument(
        "--formats",
        type=str,
        default="gif,png",
        help="Comma-separated: gif, png, or both.",
    )
    p.add_argument("--fps", type=float, default=8.0, help="GIF frame rate.")
    p.add_argument("--stride", type=int, default=4, help="Subsample future steps in the GIF.")
    p.add_argument(
        "--figsize",
        type=_parse_figsize,
        default=(10.0, 10.0),
        help="Figure size in inches as W,H or WxH (default 10,10). Larger = bigger output images.",
    )
    p.add_argument("--dpi", type=int, default=115, help="Figure DPI for GIFs; PNGs use dpi+24.")
    p.add_argument(
        "--margin_m",
        type=float,
        default=30.0,
        help="Half-width/height of the square BEV window in meters (smaller = more zoom).",
    )
    p.add_argument("--topk_modes", type=int, default=6, help="Number of highest-score modes to draw.")
    p.add_argument(
        "--only_object_types",
        type=str,
        default=None,
        help="Optional comma-separated filter, e.g. TYPE_VEHICLE,TYPE_CYCLIST.",
    )
    p.add_argument(
        "--combined_scene_output",
        type=str,
        default="none",
        choices=("none", "also", "only"),
        help=(
            "Multi-agent on one BEV: eval pickles store pred_trajs in world frame "
            "(model is agent-centric; dataset rotates/translates to global xy). "
            "'also' adds scene_<id>_multi.gif/png alongside per-agent files; "
            "'only' writes only those combined outputs."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.result_pkl):
        print(f"Missing result file: {args.result_pkl}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.scenes_dir):
        print(f"Missing scenes directory: {args.scenes_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.result_pkl)), "motion_vis")
    os.makedirs(out_dir, exist_ok=True)

    with open(args.result_pkl, "rb") as f:
        pred_dicts = pickle.load(f)

    scenes = waymo_motion_vis.group_predictions_by_scene(pred_dicts)
    if scenes:
        counts = [len(s) for s in scenes]
        mx = max(counts)
        if args.max_agents_per_scene > mx:
            print(
                f"Note: --max_agents_per_scene={args.max_agents_per_scene} but this pickle "
                f"has at most {mx} prediction(s) per scenario (min {min(counts)}, "
                f"mean {sum(counts) / len(counts):.2f}). "
                "That matches how many center agents eval ran per scene (Waymo val/test "
                "typically annotates two tracks_to_predict).",
                file=sys.stderr,
            )
    want_gif = "gif" in {x.strip().lower() for x in args.formats.split(",")}
    want_png = "png" in {x.strip().lower() for x in args.formats.split(",")}
    type_allow = None
    if args.only_object_types:
        type_allow = {x.strip() for x in args.only_object_types.split(",") if x.strip()}

    n_done = 0
    for si, agents in enumerate(scenes[: args.max_scenes]):
        if not agents:
            continue
        sid = waymo_motion_vis._to_scalar_id(agents[0]["scenario_id"])
        try:
            scene = waymo_motion_vis.load_scene_pickle(args.scenes_dir, sid)
        except FileNotFoundError:
            print(f"Skip scenario {sid}: no sample file under {args.scenes_dir}", file=sys.stderr)
            continue

        selected = []
        for pred in agents[: args.max_agents_per_scene]:
            if type_allow is not None and str(pred.get("object_type", "")) not in type_allow:
                continue
            selected.append(pred)

        comb = args.combined_scene_output
        do_per_agent = comb != "only"
        do_combined = comb in ("also", "only") and len(selected) > 0

        if do_combined:
            base_multi = f"scene_{sid}_multi"
            try:
                if want_gif:
                    gif_path = os.path.join(out_dir, f"{base_multi}.gif")
                    waymo_motion_vis.render_scene_multi_agent_gif(
                        scene,
                        selected,
                        gif_path,
                        topk_modes=args.topk_modes,
                        future_frame_stride=args.stride,
                        figsize=args.figsize,
                        dpi=args.dpi,
                        margin_m=args.margin_m,
                        fps=args.fps,
                    )
                    print(f"Wrote {gif_path}")
                if want_png:
                    png_path = os.path.join(out_dir, f"{base_multi}.png")
                    waymo_motion_vis.render_scene_multi_agent_png(
                        scene,
                        selected,
                        png_path,
                        topk_modes=args.topk_modes,
                        figsize=args.figsize,
                        dpi=args.dpi + 24,
                        margin_m=args.margin_m,
                    )
                    print(f"Wrote {png_path}")
                n_done += 1
            except Exception as exc:  # noqa: BLE001 — user-facing batch tool.
                print(f"Failed {base_multi}: {exc}", file=sys.stderr)

        if not do_per_agent:
            continue

        for pred in selected:
            oid = int(np.asarray(pred["object_id"]).reshape(-1)[0])
            base = f"scene_{sid}_agent_{oid}"
            try:
                if want_gif:
                    gif_path = os.path.join(out_dir, f"{base}.gif")
                    waymo_motion_vis.render_scene_agent_gif(
                        scene,
                        pred,
                        gif_path,
                        topk_modes=args.topk_modes,
                        future_frame_stride=args.stride,
                        figsize=args.figsize,
                        dpi=args.dpi,
                        margin_m=args.margin_m,
                        fps=args.fps,
                    )
                    print(f"Wrote {gif_path}")
                if want_png:
                    png_path = os.path.join(out_dir, f"{base}.png")
                    waymo_motion_vis.render_scene_agent_png(
                        scene,
                        pred,
                        png_path,
                        topk_modes=args.topk_modes,
                        figsize=args.figsize,
                        dpi=args.dpi + 24,
                        margin_m=args.margin_m,
                    )
                    print(f"Wrote {png_path}")
                n_done += 1
            except Exception as exc:  # noqa: BLE001 — user-facing batch tool.
                print(f"Failed {base}: {exc}", file=sys.stderr)

    if n_done == 0:
        print("No visualizations were written.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
