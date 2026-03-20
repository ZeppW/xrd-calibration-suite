from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyFAI

from poni_grid_search_4d import build_ai_from_fit2d, evaluate_one_combo
from poni_theta_phi_check import (
    parse_mask_transform,
    resolve_mask_for_shape,
    select_top_frames_from_pt_map,
)
from xrd_cali import load_image_h5


def log(msg: str):
    print(msg, flush=True)


def load_selection_and_cache(
    root_dir: Path,
    pt_map_path: Path,
    dataset: str,
    top_n: int,
):
    selected = select_top_frames_from_pt_map(
        root_dir,
        pt_map_path,
        top_n=int(top_n),
        skip_first_col=True,
    )
    image_cache: dict[str, np.ndarray] = {}
    first_shape = None
    for _, r in selected.iterrows():
        file_name = str(r["file"])
        img = load_image_h5(root_dir / file_name, dataset=dataset, frame=0, downsample=1)
        image_cache[file_name] = img
        if first_shape is None:
            first_shape = img.shape
    return selected, image_cache, first_shape


def evaluate_delta(
    dx: float,
    dy: float,
    dtilt: float,
    dtiltplan: float,
    base_ai,
    base_fit: dict,
    selected: pd.DataFrame,
    image_cache: dict[str, np.ndarray],
    dataset: str,
    mask_arr: np.ndarray | None,
    args,
) -> dict:
    ai = build_ai_from_fit2d(
        base_ai,
        base_fit,
        dx=float(dx),
        dy=float(dy),
        dtilt=float(dtilt),
        dtiltplan=float(dtiltplan),
    )
    fit2d = ai.getFit2D()
    stats = evaluate_one_combo(
        ai=ai,
        selected=selected,
        image_cache=image_cache,
        dataset=dataset,
        mask_arr=mask_arr,
        theta_guess_deg=float(args.theta_guess_deg),
        theta_guess_mode=str(args.theta_guess_mode),
        theta_guess_topk_peaks=int(args.theta_guess_topk_peaks),
        theta_search_half_window_deg=float(args.theta_search_half_window_deg),
        theta_half_window_deg=float(args.theta_half_window_deg),
        fit_row_log10_threshold=float(args.fit_row_log10_threshold),
        fit_max_abs_dev_from_theta0=float(args.fit_max_abs_dev_from_theta0),
        fit_max_abs_dev_from_file_median=float(args.fit_max_abs_dev_from_file_median),
        npt_1d=int(args.npt_1d),
        npt_rad=int(args.npt_rad),
        npt_azim=int(args.npt_azim),
        unit=str(args.unit),
    )
    amp = float(stats.get("amplitude", np.nan))
    abs_amp = float(abs(amp)) if np.isfinite(amp) else float("inf")
    return {
        "dx": float(dx),
        "dy": float(dy),
        "dtilt": float(dtilt),
        "dtiltplan": float(dtiltplan),
        "centerX": float(fit2d["centerX"]),
        "centerY": float(fit2d["centerY"]),
        "tilt": float(fit2d["tilt"]),
        "tiltPlanRotation": float(fit2d["tiltPlanRotation"]),
        "status": str(stats.get("status", "ok")),
        "amplitude": amp,
        "abs_amplitude": abs_amp,
        "offset": float(stats.get("offset", np.nan)),
        "phase_deg": float(stats.get("phase_deg", np.nan)),
        "r2": float(stats.get("r2", np.nan)),
        "rmse": float(stats.get("rmse", np.nan)),
        "n_rows_valid": int(stats.get("n_rows_valid", 0)),
        "n_joint_points": int(stats.get("n_joint_points", 0)),
        "n_fit_points": int(stats.get("n_fit_points", 0)),
    }


def run(args):
    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_map_path = Path(args.pt_map_path) if args.pt_map_path else (root_dir / "hdf5_images_output" / "maps" / "map_pt_roi_power.npy")

    base_ai = pyFAI.load(str(args.base_poni))
    base_fit = base_ai.getFit2D()
    mask_transform = parse_mask_transform(args.mask_transform)

    log(f"[1/4] Loading top-{int(args.top_n)} selection and images (col=0 excluded).")
    selected, image_cache, first_shape = load_selection_and_cache(
        root_dir=root_dir,
        pt_map_path=pt_map_path,
        dataset=str(args.dataset),
        top_n=int(args.top_n),
    )
    mask_arr = None
    if args.mask_source:
        mask_arr, resolved = resolve_mask_for_shape(
            args.mask_source,
            img_shape=first_shape,
            mask_npz_key=args.mask_npz_key,
            mask_transform=mask_transform,
            mask_is_keep_region=bool(args.mask_is_keep_region),
        )
        log(f"[1/4] Resolved mask: {resolved}")

    eval_cache: dict[tuple[float, float, float, float], dict] = {}
    history: list[dict] = []

    def eval_with_cache(dx: float, dy: float, dtilt: float, dtiltplan: float, note: str, iteration: int):
        key = (round(float(dx), 9), round(float(dy), 9), round(float(dtilt), 9), round(float(dtiltplan), 9))
        t0 = time.time()
        if key in eval_cache:
            row = dict(eval_cache[key])
            row["from_cache"] = True
        else:
            row = evaluate_delta(
                dx=dx,
                dy=dy,
                dtilt=dtilt,
                dtiltplan=dtiltplan,
                base_ai=base_ai,
                base_fit=base_fit,
                selected=selected,
                image_cache=image_cache,
                dataset=str(args.dataset),
                mask_arr=mask_arr,
                args=args,
            )
            eval_cache[key] = dict(row)
            row["from_cache"] = False
        row["note"] = str(note)
        row["iteration"] = int(iteration)
        row["elapsed_sec"] = float(time.time() - t0)
        history.append(dict(row))
        return row

    names = ["dx", "dy", "dtilt", "dtiltplan"]
    x = np.array(
        [
            float(args.start_dx),
            float(args.start_dy),
            float(args.start_dtilt),
            float(args.start_dtiltplan),
        ],
        dtype=np.float64,
    )
    steps = np.array(
        [
            float(args.step_dx),
            float(args.step_dy),
            float(args.step_dtilt),
            float(args.step_dtiltplan),
        ],
        dtype=np.float64,
    )
    min_steps = np.array(
        [
            float(args.min_step_dx),
            float(args.min_step_dy),
            float(args.min_step_dtilt),
            float(args.min_step_dtiltplan),
        ],
        dtype=np.float64,
    )

    log("[2/4] Starting local coordinate-descent refinement.")
    best = eval_with_cache(x[0], x[1], x[2], x[3], note="init", iteration=0)
    log(f"[2/4] init |amp|={best['abs_amplitude']:.8g}")

    for it in range(1, int(args.max_iters) + 1):
        improved = False
        log(
            f"[2/4] iter={it}, steps=({steps[0]:.5g},{steps[1]:.5g},{steps[2]:.5g},{steps[3]:.5g}), "
            f"best |amp|={best['abs_amplitude']:.8g}"
        )
        for j, name in enumerate(names):
            for sign in (+1.0, -1.0):
                cand = x.copy()
                cand[j] += sign * steps[j]
                row = eval_with_cache(cand[0], cand[1], cand[2], cand[3], note=f"probe {name} {sign:+.0f}", iteration=it)
                if row["abs_amplitude"] + float(args.improve_tol) < best["abs_amplitude"]:
                    x = cand
                    best = row
                    improved = True
                    log(
                        f"  improved -> |amp|={best['abs_amplitude']:.8g} "
                        f"at dx={x[0]:+.6f},dy={x[1]:+.6f},dtilt={x[2]:+.6f},dtiltplan={x[3]:+.6f}"
                    )
        if not improved:
            steps *= float(args.step_shrink)
            if np.all(steps <= min_steps):
                log("[2/4] Stopping: all step sizes reached minimum.")
                break

    best_ai = build_ai_from_fit2d(
        base_ai,
        base_fit,
        dx=float(best["dx"]),
        dy=float(best["dy"]),
        dtilt=float(best["dtilt"]),
        dtiltplan=float(best["dtiltplan"]),
    )
    best_poni_path = out_dir / "best_local_candidate.poni"
    best_ai.save(str(best_poni_path))

    # Essential validation pass on a larger top-N (e.g. 20) without using it
    # for every single local step.
    validation = {
        "performed": False,
    }
    if int(args.top_n_essential) > int(args.top_n):
        log(f"[3/4] Essential validation on top-{int(args.top_n_essential)}.")
        sel2, cache2, first_shape2 = load_selection_and_cache(
            root_dir=root_dir,
            pt_map_path=pt_map_path,
            dataset=str(args.dataset),
            top_n=int(args.top_n_essential),
        )
        mask_arr2 = None
        if args.mask_source:
            mask_arr2, _ = resolve_mask_for_shape(
                args.mask_source,
                img_shape=first_shape2,
                mask_npz_key=args.mask_npz_key,
                mask_transform=mask_transform,
                mask_is_keep_region=bool(args.mask_is_keep_region),
            )

        best_topn = evaluate_delta(
            dx=float(best["dx"]),
            dy=float(best["dy"]),
            dtilt=float(best["dtilt"]),
            dtiltplan=float(best["dtiltplan"]),
            base_ai=base_ai,
            base_fit=base_fit,
            selected=sel2,
            image_cache=cache2,
            dataset=str(args.dataset),
            mask_arr=mask_arr2,
            args=args,
        )
        base_topn = evaluate_delta(
            dx=0.0,
            dy=0.0,
            dtilt=0.0,
            dtiltplan=0.0,
            base_ai=base_ai,
            base_fit=base_fit,
            selected=sel2,
            image_cache=cache2,
            dataset=str(args.dataset),
            mask_arr=mask_arr2,
            args=args,
        )
        validation = {
            "performed": True,
            "top_n_essential": int(args.top_n_essential),
            "best_on_essential_topn": best_topn,
            "baseline_on_essential_topn": base_topn,
            "abs_amp_improvement": float(base_topn["abs_amplitude"] - best_topn["abs_amplitude"]),
        }

    log("[4/4] Writing outputs.")
    history_df = pd.DataFrame(history)
    evals_df = pd.DataFrame([v for v in eval_cache.values()]).sort_values("abs_amplitude").reset_index(drop=True)
    history_df.to_csv(out_dir / "local_refine_history.csv", index=False)
    evals_df.to_csv(out_dir / "local_refine_unique_evals.csv", index=False)

    summary = {
        "settings": {
            "root_dir": str(root_dir),
            "pt_map_path": str(pt_map_path),
            "base_poni": str(args.base_poni),
            "top_n": int(args.top_n),
            "top_n_essential": int(args.top_n_essential),
            "selection_skip_first_col": True,
            "directDist_locked": True,
        },
        "best_local": best,
        "n_evals_total": int(len(history_df)),
        "n_unique_evals": int(len(evals_df)),
        "validation": validation,
        "files": {
            "best_local_candidate_poni": str(best_poni_path),
            "local_refine_history_csv": str(out_dir / "local_refine_history.csv"),
            "local_refine_unique_evals_csv": str(out_dir / "local_refine_unique_evals.csv"),
        },
    }
    (out_dir / "local_refine_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Local gradient-like (coordinate descent) refinement for PONI deltas. "
            "Optimizes on top-N (fast), then optionally validates on top-N-essential (e.g. 20). "
            "directDist is kept fixed."
        )
    )
    p.add_argument("--base-poni", type=str, default=r"E:\XRD\proc\proc\LaB6_003_25keV_poni.poni")
    p.add_argument("--root-dir", type=str, default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files")
    p.add_argument("--dataset", type=str, default="data")
    p.add_argument("--pt-map-path", type=str, default=None)
    p.add_argument("--mask-source", type=str, default=r"F:\NMR\NMR\py_projects\xrd\annulus_phi_mask.mask")
    p.add_argument("--mask-npz-key", type=str, default=None)
    p.add_argument("--mask-transform", type=str, default="flipud")
    p.add_argument("--mask-is-keep-region", action="store_true")

    p.add_argument("--start-dx", type=float, default=0.0)
    p.add_argument("--start-dy", type=float, default=0.0)
    p.add_argument("--start-dtilt", type=float, default=0.0)
    p.add_argument("--start-dtiltplan", type=float, default=0.0)

    p.add_argument("--step-dx", type=float, default=0.2)
    p.add_argument("--step-dy", type=float, default=0.1)
    p.add_argument("--step-dtilt", type=float, default=0.04)
    p.add_argument("--step-dtiltplan", type=float, default=0.5)

    p.add_argument("--min-step-dx", type=float, default=0.01)
    p.add_argument("--min-step-dy", type=float, default=0.01)
    p.add_argument("--min-step-dtilt", type=float, default=0.005)
    p.add_argument("--min-step-dtiltplan", type=float, default=0.05)

    p.add_argument("--max-iters", type=int, default=15)
    p.add_argument("--step-shrink", type=float, default=0.5)
    p.add_argument("--improve-tol", type=float, default=1e-10)

    p.add_argument("--top-n", type=int, default=10, help="Fast optimization selection size.")
    p.add_argument(
        "--top-n-essential",
        type=int,
        default=20,
        help="Optional larger selection size for an essential validation pass (not every iteration).",
    )

    p.add_argument("--unit", type=str, default="2th_deg")
    p.add_argument("--npt-1d", type=int, default=5000)
    p.add_argument("--npt-rad", type=int, default=1200)
    p.add_argument("--npt-azim", type=int, default=720)
    p.add_argument("--theta-guess-deg", type=float, default=12.8)
    p.add_argument("--theta-guess-mode", type=str, default="strong_peaks", choices=["fixed", "strong_peaks"])
    p.add_argument("--theta-guess-topk-peaks", type=int, default=3)
    p.add_argument("--theta-search-half-window-deg", type=float, default=0.8)
    p.add_argument("--theta-half-window-deg", type=float, default=0.5)
    p.add_argument("--fit-row-log10-threshold", type=float, default=2.5)
    p.add_argument("--fit-max-abs-dev-from-theta0", type=float, default=0.2)
    p.add_argument("--fit-max-abs-dev-from-file-median", type=float, default=0.12)

    p.add_argument(
        "--out-dir",
        type=str,
        default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files\hdf5_images_output\poni_local_refine",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
