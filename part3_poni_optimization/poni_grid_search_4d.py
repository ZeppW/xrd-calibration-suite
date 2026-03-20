from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyFAI
from scipy.optimize import curve_fit

from poni_theta_phi_check import (
    estimate_theta_guess_from_strong_peaks,
    find_theta0_local_minimum,
    fit_theta_vs_phi,
    integrate_1d_theta,
    parse_mask_transform,
    resolve_mask_for_shape,
    select_top_frames_from_pt_map,
)
from theta_phi_joint_sine_fit import sine_model
from xrd_cali import integrate_cake_2d, load_image_h5


def log(msg: str):
    print(msg, flush=True)


def parse_float_list(text: str) -> list[float]:
    vals = []
    for s in str(text).split(","):
        s = s.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError(f"No values parsed from: {text}")
    return vals


def ensure_csv_header(path: Path, fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()


def append_csv_row(path: Path, fields: list[str], row: dict):
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writerow({k: row.get(k, "") for k in fields})


def fit_joint_sine_from_theta_phi(df_all_valid: pd.DataFrame, min_count: int = 2) -> dict:
    if df_all_valid.empty:
        return {
            "status": "no_valid_rows",
            "n_rows_valid": 0,
            "n_joint_points": 0,
            "n_fit_points": 0,
            "offset": np.nan,
            "amplitude": np.nan,
            "phase_deg": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
        }

    df = df_all_valid.copy()
    means = df.groupby("file")["theta_center_deg"].mean().rename("theta_mean_file")
    df = df.join(means, on="file")
    df["theta_norm"] = df["theta_center_deg"] / df["theta_mean_file"]

    joint = (
        df.groupby("phi_deg", as_index=False)
        .agg(theta_norm_mean=("theta_norm", "mean"), n=("theta_norm", "size"))
        .sort_values("phi_deg")
        .reset_index(drop=True)
    )
    fit_df = joint[joint["n"] >= int(min_count)].copy()
    if len(fit_df) < 6:
        return {
            "status": "not_enough_fit_points",
            "n_rows_valid": int(len(df)),
            "n_joint_points": int(len(joint)),
            "n_fit_points": int(len(fit_df)),
            "offset": np.nan,
            "amplitude": np.nan,
            "phase_deg": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
        }

    x = fit_df["phi_deg"].to_numpy(dtype=np.float64)
    y = fit_df["theta_norm_mean"].to_numpy(dtype=np.float64)
    w = fit_df["n"].to_numpy(dtype=np.float64)

    p0 = [float(np.mean(y)), float((np.max(y) - np.min(y)) * 0.5), 0.0]
    sigma = 1.0 / np.sqrt(np.maximum(w, 1.0))
    popt, _ = curve_fit(
        sine_model,
        x,
        y,
        p0=p0,
        sigma=sigma,
        absolute_sigma=False,
        maxfev=20000,
    )
    y_fit = sine_model(x, *popt)
    resid = y - y_fit
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid * resid)))

    return {
        "status": "ok",
        "n_rows_valid": int(len(df)),
        "n_joint_points": int(len(joint)),
        "n_fit_points": int(len(fit_df)),
        "offset": float(popt[0]),
        "amplitude": float(popt[1]),
        "phase_deg": float(popt[2]),
        "r2": r2,
        "rmse": rmse,
    }


def build_ai_from_fit2d(base_ai, base_fit2d: dict, dx: float, dy: float, dtilt: float, dtiltplan: float):
    ai = copy.deepcopy(base_ai)
    ai.setFit2D(
        directDist=float(base_fit2d["directDist"]),
        centerX=float(base_fit2d["centerX"]) + float(dx),
        centerY=float(base_fit2d["centerY"]) + float(dy),
        tilt=float(base_fit2d["tilt"]) + float(dtilt),
        tiltPlanRotation=float(base_fit2d["tiltPlanRotation"]) + float(dtiltplan),
        pixelX=float(base_fit2d["pixelX"]),
        pixelY=float(base_fit2d["pixelY"]),
        splinefile=base_fit2d.get("splinefile", None),
        wavelength=base_fit2d.get("wavelength", None),
    )
    return ai


def evaluate_one_combo(
    ai,
    selected: pd.DataFrame,
    image_cache: dict[str, np.ndarray],
    dataset: str,
    mask_arr: np.ndarray | None,
    theta_guess_deg: float,
    theta_guess_mode: str,
    theta_guess_topk_peaks: int,
    theta_search_half_window_deg: float,
    theta_half_window_deg: float,
    fit_row_log10_threshold: float,
    fit_max_abs_dev_from_theta0: float,
    fit_max_abs_dev_from_file_median: float,
    npt_1d: int,
    npt_rad: int,
    npt_azim: int,
    unit: str,
) -> dict:
    all_rows = []

    for _, r in selected.iterrows():
        file_name = str(r["file"])
        img = image_cache[file_name]

        tth1d, i1d = integrate_1d_theta(ai, img, mask=mask_arr, npt=npt_1d, unit=unit)

        guess_file = float(theta_guess_deg)
        if str(theta_guess_mode).lower() == "strong_peaks":
            guess_file, _ = estimate_theta_guess_from_strong_peaks(
                tth1d,
                i1d,
                center_hint_deg=float(theta_guess_deg),
                half_window_deg=float(theta_search_half_window_deg),
                topk=int(theta_guess_topk_peaks),
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
            all_rows.append(df_fp[["file", "phi_deg", "theta_center_deg"]])

    if all_rows:
        df_all_valid = pd.concat(all_rows, axis=0, ignore_index=True)
    else:
        df_all_valid = pd.DataFrame(columns=["file", "phi_deg", "theta_center_deg"])
    return fit_joint_sine_from_theta_phi(df_all_valid, min_count=2)


def run(args):
    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "grid_results.csv"
    best_json = out_dir / "best_result.json"
    best_poni = out_dir / "best_candidate.poni"

    dx_vals = parse_float_list(args.dx_values)
    dy_vals = parse_float_list(args.dy_values)
    dtilt_vals = parse_float_list(args.dtilt_values)
    dtiltplan_vals = parse_float_list(args.dtiltplan_values)
    combos = list(itertools.product(dx_vals, dy_vals, dtilt_vals, dtiltplan_vals))

    fields = [
        "trial_index",
        "dx",
        "dy",
        "dtilt",
        "dtiltplan",
        "centerX",
        "centerY",
        "tilt",
        "tiltPlanRotation",
        "status",
        "n_rows_valid",
        "n_joint_points",
        "n_fit_points",
        "offset",
        "amplitude",
        "phase_deg",
        "r2",
        "rmse",
        "elapsed_sec",
        "error",
    ]
    ensure_csv_header(results_csv, fields)

    done = set()
    if results_csv.exists():
        try:
            old = pd.read_csv(results_csv)
            for _, r in old.iterrows():
                done.add((float(r["dx"]), float(r["dy"]), float(r["dtilt"]), float(r["dtiltplan"])))
        except Exception:
            pass

    log(f"Total combos: {len(combos)}")
    log(f"Already done: {len(done)}")

    base_ai = pyFAI.load(str(args.base_poni))
    base_fit = base_ai.getFit2D()
    mask_transform = parse_mask_transform(args.mask_transform)

    pt_map_path = Path(args.pt_map_path) if args.pt_map_path else (root_dir / "hdf5_images_output" / "maps" / "map_pt_roi_power.npy")
    selected = select_top_frames_from_pt_map(
        root_dir,
        pt_map_path,
        top_n=int(args.top_n),
        skip_first_col=True,
    )
    log(f"Selected top {len(selected)} files from Pt map (excluding col=0): {pt_map_path}")

    image_cache: dict[str, np.ndarray] = {}
    first_shape = None
    for _, r in selected.iterrows():
        file_name = str(r["file"])
        path_h5 = root_dir / file_name
        img = load_image_h5(path_h5, dataset=args.dataset, frame=0, downsample=1)
        image_cache[file_name] = img
        if first_shape is None:
            first_shape = img.shape
    log(f"Loaded {len(image_cache)} images into cache.")

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

    best_amp = np.inf
    best_row = None

    start_all = time.time()
    for idx, (dx, dy, dtilt, dtiltplan) in enumerate(combos, start=1):
        key = (float(dx), float(dy), float(dtilt), float(dtiltplan))
        if key in done:
            continue

        t0 = time.time()
        log(
            f"[{idx}/{len(combos)}] dx={dx:+.3f}, dy={dy:+.3f}, dtilt={dtilt:+.4f}, "
            f"dtiltplan={dtiltplan:+.4f}"
        )
        row = {
            "trial_index": idx,
            "dx": float(dx),
            "dy": float(dy),
            "dtilt": float(dtilt),
            "dtiltplan": float(dtiltplan),
            "status": "error",
            "error": "",
        }
        try:
            ai = build_ai_from_fit2d(base_ai, base_fit, dx=dx, dy=dy, dtilt=dtilt, dtiltplan=dtiltplan)
            fit = ai.getFit2D()
            row["centerX"] = float(fit["centerX"])
            row["centerY"] = float(fit["centerY"])
            row["tilt"] = float(fit["tilt"])
            row["tiltPlanRotation"] = float(fit["tiltPlanRotation"])

            stats = evaluate_one_combo(
                ai=ai,
                selected=selected,
                image_cache=image_cache,
                dataset=args.dataset,
                mask_arr=mask_arr,
                theta_guess_deg=float(args.theta_guess_deg),
                theta_guess_mode=args.theta_guess_mode,
                theta_guess_topk_peaks=int(args.theta_guess_topk_peaks),
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
            row.update(stats)
            row["status"] = str(stats.get("status", "ok"))
            amp = float(row.get("amplitude", np.nan))
            if np.isfinite(amp) and abs(amp) < best_amp:
                best_amp = abs(amp)
                best_row = dict(row)
                # pyFAI.save can append if file exists; force overwrite to keep a single calibration block.
                if best_poni.exists():
                    best_poni.unlink()
                ai.save(str(best_poni))
                best_json.write_text(json.dumps(best_row, indent=2), encoding="utf-8")
                log(f"  New best |amp|={best_amp:.8g} at dx={dx:+.3f},dy={dy:+.3f},dtilt={dtilt:+.4f},dtiltplan={dtiltplan:+.4f}")
        except Exception as e:
            row["error"] = repr(e)

        row["elapsed_sec"] = float(time.time() - t0)
        append_csv_row(results_csv, fields, row)
        done.add(key)

    total_sec = time.time() - start_all
    log(f"Finished grid search in {total_sec/60.0:.1f} min")
    if best_row is not None:
        log("Best result:")
        log(json.dumps(best_row, indent=2))
    else:
        log("No valid result found.")


def build_parser():
    p = argparse.ArgumentParser(description="Small 4D grid search on Fit2D params and sine amplitude.")
    p.add_argument("--base-poni", type=str, default=r"E:\XRD\proc\proc\LaB6_003_25keV_poni.poni")
    p.add_argument("--root-dir", type=str, default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files")
    p.add_argument("--dataset", type=str, default="data")
    p.add_argument("--pt-map-path", type=str, default=None)
    p.add_argument("--mask-source", type=str, default=r"F:\NMR\NMR\py_projects\xrd\annulus_phi_mask.mask")
    p.add_argument("--mask-npz-key", type=str, default=None)
    p.add_argument("--mask-transform", type=str, default="flipud")
    p.add_argument("--mask-is-keep-region", action="store_true")
    p.add_argument("--top-n", type=int, default=10)

    p.add_argument("--dx-values", type=str, default="-1,0,1")
    p.add_argument("--dy-values", type=str, default="-1,0,1")
    p.add_argument("--dtilt-values", type=str, default="-0.1,0,0.1")
    p.add_argument("--dtiltplan-values", type=str, default="-6,0,6")

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
        default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files\hdf5_images_output\poni_grid_search_4d",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
