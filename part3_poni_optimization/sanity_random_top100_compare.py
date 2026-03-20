from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyFAI

from compare_poni_merged_centered_fit import fit_merged_sine, plot_comparison
from poni_theta_phi_check import (
    estimate_theta_guess_from_strong_peaks,
    find_theta0_local_minimum,
    fit_theta_vs_phi,
    integrate_1d_theta,
    parse_mask_transform,
    resolve_mask_for_shape,
    select_top_frames_from_pt_map,
)
from xrd_cali import integrate_cake_2d, load_image_h5


def log(msg: str):
    print(msg, flush=True)


def build_centered_merged(df_valid: pd.DataFrame) -> pd.DataFrame:
    if df_valid.empty:
        return pd.DataFrame(columns=["key", "frame_index", "phi_deg", "theta_center_deg", "theta_mean_file", "theta_centered_deg"])
    out = df_valid.copy()
    means = out.groupby("file")["theta_center_deg"].mean().rename("theta_mean_file")
    out = out.join(means, on="file")
    out["theta_centered_deg"] = out["theta_center_deg"] - out["theta_mean_file"]
    out["key"] = out["file"].map(lambda s: Path(str(s)).stem)
    out["frame_index"] = out["key"].str.extract(r"_(\d+)_restored", expand=False).astype(float).fillna(-1).astype(int)
    return out[["key", "frame_index", "phi_deg", "theta_center_deg", "theta_mean_file", "theta_centered_deg", "file"]]


def evaluate_subset_for_poni(
    subset_df: pd.DataFrame,
    image_cache: dict[str, np.ndarray],
    ai,
    mask_arr: np.ndarray | None,
    theta_guess_deg: float,
    theta_search_half_window_deg: float,
    theta_half_window_deg: float,
    fit_row_log10_threshold: float,
    fit_max_abs_dev_from_theta0: float,
    fit_max_abs_dev_from_file_median: float,
    npt_1d: int,
    npt_rad: int,
    npt_azim: int,
    unit: str,
) -> pd.DataFrame:
    rows = []
    for _, r in subset_df.iterrows():
        file_name = str(r["file"])
        img = image_cache[file_name]

        tth1d, i1d = integrate_1d_theta(ai, img, mask=mask_arr, npt=npt_1d, unit=unit)
        guess_file, _ = estimate_theta_guess_from_strong_peaks(
            tth1d,
            i1d,
            center_hint_deg=float(theta_guess_deg),
            half_window_deg=float(theta_search_half_window_deg),
            topk=3,
        )
        theta0, info = find_theta0_local_minimum(
            tth1d,
            i1d,
            guess_deg=guess_file,
            half_window_deg=float(theta_search_half_window_deg),
        )
        if abs(float(theta0) - float(guess_file)) > 0.18:
            peak_theta = info.get("peak_theta_deg", None)
            if peak_theta is not None and np.isfinite(peak_theta):
                theta0 = float(peak_theta)

        theta_lo = float(theta0) - float(theta_half_window_deg)
        theta_hi = float(theta0) + float(theta_half_window_deg)
        cake, tth_axis, phi_axis = integrate_cake_2d(
            img,
            ai=ai,
            npt_rad=int(npt_rad),
            npt_azim=int(npt_azim),
            unit=unit,
            radial_range=(theta_lo, theta_hi),
            mask=mask_arr,
            azimuth_to_0_360=True,
            return_info=False,
        )
        df_fp = fit_theta_vs_phi(
            cake,
            tth_axis,
            phi_axis,
            theta_guess=float(theta0),
            min_row_log10_peak=float(fit_row_log10_threshold),
            max_abs_dev_from_theta0=float(fit_max_abs_dev_from_theta0),
            max_abs_dev_from_file_median=float(fit_max_abs_dev_from_file_median),
        )
        df_fp["file"] = file_name
        df_fp = df_fp[np.isfinite(df_fp["theta_center_deg"])].copy()
        if not df_fp.empty:
            rows.append(df_fp[["file", "phi_deg", "theta_center_deg"]])

    if rows:
        return pd.concat(rows, axis=0, ignore_index=True)
    return pd.DataFrame(columns=["file", "phi_deg", "theta_center_deg"])


def run(args):
    root_dir = Path(args.root_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed) if args.seed is not None else int(time.time()) % 1_000_000_000
    rng = np.random.default_rng(seed)

    top100 = select_top_frames_from_pt_map(
        root_dir,
        Path(args.pt_map_path),
        top_n=100,
        skip_first_col=True,
    )
    if len(top100) < 10:
        raise RuntimeError(f"Need at least 10 points in top100 selection, got {len(top100)}")

    excl = int(args.exclude_top_n)
    if excl < 0:
        raise ValueError("--exclude-top-n must be >= 0")
    pool = top100.iloc[excl:].reset_index(drop=True)
    if len(pool) < 10:
        raise RuntimeError(
            f"Not enough points after excluding top {excl}. "
            f"Pool size={len(pool)} from top100."
        )

    pick_idx = rng.choice(len(pool), size=10, replace=False)
    subset = pool.iloc[np.sort(pick_idx)].reset_index(drop=True)
    subset.to_csv(out_root / "random_top100_subset.csv", index=False)

    log(f"Random seed: {seed}")
    log(f"Excluded top-{excl} highest Pt points from sampling pool.")
    log("Point preselection excludes first column (col=0).")
    log("Selected files:")
    log(subset[["file", "row", "col", "pt_roi_power_log10"]].to_string(index=False))

    # Cache images once.
    image_cache: dict[str, np.ndarray] = {}
    first_shape = None
    for _, r in subset.iterrows():
        file_name = str(r["file"])
        img = load_image_h5(root_dir / file_name, dataset=args.dataset, frame=0, downsample=1)
        image_cache[file_name] = img
        if first_shape is None:
            first_shape = img.shape

    mask_transform = parse_mask_transform(args.mask_transform)
    mask_arr = None
    if args.mask_source:
        mask_arr, resolved_mask = resolve_mask_for_shape(
            args.mask_source,
            img_shape=first_shape,
            mask_npz_key=args.mask_npz_key,
            mask_transform=mask_transform,
            mask_is_keep_region=bool(args.mask_is_keep_region),
        )
        log(f"Resolved mask: {resolved_mask}")

    ai_orig = pyFAI.load(str(args.poni_original))
    ai_best = pyFAI.load(str(args.poni_best))

    log("Evaluating original PONI on subset...")
    df_valid_orig = evaluate_subset_for_poni(
        subset,
        image_cache=image_cache,
        ai=ai_orig,
        mask_arr=mask_arr,
        theta_guess_deg=float(args.theta_guess_deg),
        theta_search_half_window_deg=float(args.theta_search_half_window_deg),
        theta_half_window_deg=float(args.theta_half_window_deg),
        fit_row_log10_threshold=float(args.fit_row_log10_threshold),
        fit_max_abs_dev_from_theta0=float(args.fit_max_abs_dev_from_theta0),
        fit_max_abs_dev_from_file_median=float(args.fit_max_abs_dev_from_file_median),
        npt_1d=int(args.npt_1d),
        npt_rad=int(args.npt_rad),
        npt_azim=int(args.npt_azim),
        unit=args.unit,
    )
    df_valid_orig.to_csv(out_root / "theta_vs_phi_all_valid_original.csv", index=False)

    log("Evaluating best PONI on subset...")
    df_valid_best = evaluate_subset_for_poni(
        subset,
        image_cache=image_cache,
        ai=ai_best,
        mask_arr=mask_arr,
        theta_guess_deg=float(args.theta_guess_deg),
        theta_search_half_window_deg=float(args.theta_search_half_window_deg),
        theta_half_window_deg=float(args.theta_half_window_deg),
        fit_row_log10_threshold=float(args.fit_row_log10_threshold),
        fit_max_abs_dev_from_theta0=float(args.fit_max_abs_dev_from_theta0),
        fit_max_abs_dev_from_file_median=float(args.fit_max_abs_dev_from_file_median),
        npt_1d=int(args.npt_1d),
        npt_rad=int(args.npt_rad),
        npt_azim=int(args.npt_azim),
        unit=args.unit,
    )
    df_valid_best.to_csv(out_root / "theta_vs_phi_all_valid_best.csv", index=False)

    merged_orig = build_centered_merged(df_valid_orig)
    merged_best = build_centered_merged(df_valid_best)
    merged_orig.to_csv(out_root / "merged_centered_original.csv", index=False)
    merged_best.to_csv(out_root / "merged_centered_best.csv", index=False)

    fit_orig = fit_merged_sine(merged_orig.rename(columns={"theta_centered_deg": "theta_centered_deg"}))
    fit_best = fit_merged_sine(merged_best.rename(columns={"theta_centered_deg": "theta_centered_deg"}))

    summary = {
        "seed": seed,
        "exclude_top_n": excl,
        "n_subset_files": int(len(subset)),
        "n_points_original": int(len(merged_orig)),
        "n_points_best": int(len(merged_best)),
        "fit_original": fit_orig,
        "fit_best": fit_best,
        "abs_amp_improvement": float(abs(fit_orig.get("amplitude", np.nan)) - abs(fit_best.get("amplitude", np.nan))),
        "best_is_better_by_abs_amp": bool(abs(fit_best.get("amplitude", np.inf)) < abs(fit_orig.get("amplitude", np.inf))),
    }
    (out_root / "sanity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Reuse existing merged comparison plotting.
    plot_comparison(
        merged_orig.rename(columns={"theta_centered_deg": "theta_centered_deg"}),
        fit_orig,
        merged_best.rename(columns={"theta_centered_deg": "theta_centered_deg"}),
        fit_best,
        out_root / "merged_centered_sine_fit_original_vs_best_random10.png",
    )

    print(f"Wrote outputs to: {out_root}")
    print(json.dumps(summary, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description="Sanity test: random 10 from top 100 Pt points, compare merged-centered sine fit.")
    p.add_argument("--root-dir", type=str, default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files")
    p.add_argument(
        "--pt-map-path",
        type=str,
        default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files\hdf5_images_output\maps\map_pt_roi_power.npy",
    )
    p.add_argument("--dataset", type=str, default="data")
    p.add_argument("--poni-original", type=str, default=r"E:\XRD\proc\proc\LaB6_003_25keV_poni.poni")
    p.add_argument(
        "--poni-best",
        type=str,
        default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files\hdf5_images_output\poni_grid_search_4d_global\global_best_candidate.poni",
    )
    p.add_argument("--mask-source", type=str, default=r"F:\NMR\NMR\py_projects\xrd\annulus_phi_mask.mask")
    p.add_argument("--mask-npz-key", type=str, default=None)
    p.add_argument("--mask-transform", type=str, default="flipud")
    p.add_argument("--mask-is-keep-region", action="store_true")

    p.add_argument("--unit", type=str, default="2th_deg")
    p.add_argument("--npt-1d", type=int, default=5000)
    p.add_argument("--npt-rad", type=int, default=1200)
    p.add_argument("--npt-azim", type=int, default=720)
    p.add_argument("--theta-guess-deg", type=float, default=12.8)
    p.add_argument("--theta-search-half-window-deg", type=float, default=0.8)
    p.add_argument("--theta-half-window-deg", type=float, default=0.5)
    p.add_argument("--fit-row-log10-threshold", type=float, default=2.5)
    p.add_argument("--fit-max-abs-dev-from-theta0", type=float, default=0.2)
    p.add_argument("--fit-max-abs-dev-from-file-median", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--exclude-top-n", type=int, default=10)
    p.add_argument(
        "--out-dir",
        type=str,
        default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files\hdf5_images_output\poni_compare\sanity_random_top100",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
