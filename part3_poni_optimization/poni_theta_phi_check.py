from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyFAI
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

from xrd_cali import (
    _resolve_mask_source,
    integrate_cake_2d,
    load_image_h5,
    plot_cake_2d,
    save_cake_npz,
)


def log(msg: str):
    print(msg, flush=True)


def frame_index_from_name(name: str) -> int:
    m = re.search(r"_(\d+)_restored", name)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else -1


def find_h5_files(root_dir: Path) -> list[Path]:
    files = [p for p in root_dir.glob("*.hdf5")]
    files.extend([p for p in root_dir.glob("*.h5") if p not in files])
    return sorted(files, key=lambda p: frame_index_from_name(p.name))


def parse_mask_transform(value: str | None):
    if value is None:
        return None
    txt = str(value).strip()
    if not txt or txt.lower() == "none":
        return None
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else parts


def apply_mask_transform(mask_arr: np.ndarray, mask_transform):
    if mask_transform is None:
        return np.asarray(mask_arr, dtype=bool)
    out = np.asarray(mask_arr, dtype=bool)
    ops = [mask_transform] if isinstance(mask_transform, str) else list(mask_transform)
    for op in ops:
        if op == "flipud":
            out = np.flipud(out)
        elif op == "fliplr":
            out = np.fliplr(out)
        elif op == "transpose":
            out = out.T
        else:
            raise ValueError(f"Unknown mask_transform op '{op}'. Use flipud/fliplr/transpose.")
    return out


def resolve_mask_for_shape(
    mask_source: str | Path | None,
    img_shape: tuple[int, int],
    mask_npz_key: str | None,
    mask_transform,
    mask_is_keep_region: bool,
):
    if mask_source is None:
        return None, None
    raw, resolved = _resolve_mask_source(mask_source, expected_shape=img_shape, npz_key=mask_npz_key)
    mask_arr = np.asarray(raw != 0, dtype=bool)
    mask_arr = apply_mask_transform(mask_arr, mask_transform)
    if mask_is_keep_region:
        mask_arr = ~mask_arr
    return mask_arr, resolved


def select_top_frames_from_pt_map(
    root_dir: Path,
    pt_map_path: Path,
    top_n: int = 10,
    skip_first_col: bool = True,
) -> pd.DataFrame:
    if not pt_map_path.exists():
        raise FileNotFoundError(f"Pt map not found: {pt_map_path}")

    pt_map = np.load(pt_map_path)
    if pt_map.ndim != 2:
        raise ValueError(f"Expected 2D Pt map, got shape {pt_map.shape} from {pt_map_path}")

    ny, nx = pt_map.shape
    files = find_h5_files(root_dir)
    if not files:
        raise RuntimeError(f"No HDF5 files found under {root_dir}")

    rows = []
    for idx, power in enumerate(pt_map.ravel(order="C")):
        if idx >= len(files):
            break
        if (not np.isfinite(power)) or (power <= 0):
            continue
        r, c = divmod(idx, nx)
        if bool(skip_first_col) and int(c) == 0:
            continue
        f = files[idx]
        rows.append(
            {
                "row": int(r),
                "col": int(c),
                "file": f.name,
                "frame_index": int(frame_index_from_name(f.name)),
                "path": str(f.resolve()),
                "pt_roi_power": float(power),
                "pt_roi_power_log10": float(np.log10(power)),
            }
        )

    if not rows:
        raise RuntimeError(f"No valid positive values in map: {pt_map_path}")

    df = pd.DataFrame(rows).sort_values("pt_roi_power_log10", ascending=False).reset_index(drop=True)
    out = df.head(int(top_n)).copy()
    out["map_shape"] = f"{ny}x{nx}"
    out["skip_first_col"] = bool(skip_first_col)
    return out


def to_1d_arrays(integrate1d_result):
    if hasattr(integrate1d_result, "radial") and hasattr(integrate1d_result, "intensity"):
        x = np.asarray(integrate1d_result.radial, dtype=np.float64)
        y = np.asarray(integrate1d_result.intensity, dtype=np.float64)
        return x, y
    if isinstance(integrate1d_result, (tuple, list)) and len(integrate1d_result) >= 2:
        a = np.asarray(integrate1d_result[0], dtype=np.float64)
        b = np.asarray(integrate1d_result[1], dtype=np.float64)
        if a.ndim == 1 and b.ndim == 1:
            if np.nanmedian(np.diff(a)) > 0:
                return a, b
            if np.nanmedian(np.diff(b)) > 0:
                return b, a
        return a, b
    raise TypeError("Unsupported integrate1d result format.")


def integrate_1d_theta(
    ai,
    img: np.ndarray,
    mask: np.ndarray | None = None,
    npt: int = 5000,
    unit: str = "2th_deg",
):
    res = ai.integrate1d(
        np.asarray(img, dtype=np.float32),
        npt=int(npt),
        unit=unit,
        mask=mask,
        correctSolidAngle=True,
        method=("bbox", "csr", "cython"),
    )
    return to_1d_arrays(res)


def smooth_1d(y: np.ndarray, max_window: int = 101) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 7:
        return y.copy()
    w = min(max_window, n if (n % 2 == 1) else (n - 1))
    if w < 5:
        return y.copy()
    poly = min(3, w - 2)
    return savgol_filter(y, window_length=w, polyorder=poly, mode="interp")


def find_theta0_local_minimum(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    guess_deg: float = 12.8,
    half_window_deg: float = 0.8,
) -> tuple[float, dict]:
    x = np.asarray(two_theta, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 20:
        raise RuntimeError("Not enough valid points in 1D profile.")

    lo = float(guess_deg - half_window_deg)
    hi = float(guess_deg + half_window_deg)
    w = (x >= lo) & (x <= hi)
    if np.count_nonzero(w) < 10:
        raise RuntimeError(f"Not enough points in search window [{lo:.3f}, {hi:.3f}] deg.")

    xs = x[w]
    ys = y[w]
    ys_s = smooth_1d(ys)

    # Robust center estimate:
    # 1) find dominant peak near guess, 2) find nearest minima on left/right,
    # 3) theta0 = average(min_left, min_right). This keeps theta0 near the ring center.
    pks, _ = find_peaks(ys_s, distance=max(1, ys_s.size // 30))
    if pks.size == 0:
        peak_idx = int(np.argmax(ys_s))
        peak_mode = "window_argmax"
    else:
        peak_idx = int(pks[np.argmin(np.abs(xs[pks] - float(guess_deg)))])
        peak_mode = "local_peak_closest_to_guess"

    mins, _ = find_peaks(-ys_s, distance=max(1, ys_s.size // 30))
    left = mins[mins < peak_idx]
    right = mins[mins > peak_idx]

    if left.size > 0 and right.size > 0:
        li = int(left[-1])
        ri = int(right[0])
        theta0 = 0.5 * (float(xs[li]) + float(xs[ri]))
        mode = "avg_left_right_minima"
    elif left.size > 0:
        li = int(left[-1])
        theta0 = 0.5 * (float(xs[li]) + float(xs[peak_idx]))
        mode = "avg_left_min_peak"
        ri = None
    elif right.size > 0:
        ri = int(right[0])
        theta0 = 0.5 * (float(xs[peak_idx]) + float(xs[ri]))
        mode = "avg_peak_right_min"
        li = None
    else:
        theta0 = float(xs[peak_idx])
        mode = "peak_only"
        li = None
        ri = None

    info = {
        "search_lo_deg": lo,
        "search_hi_deg": hi,
        "method": mode,
        "peak_method": peak_mode,
        "peak_theta_deg": float(xs[peak_idx]),
        "left_min_theta_deg": (float(xs[li]) if li is not None else None),
        "right_min_theta_deg": (float(xs[ri]) if ri is not None else None),
    }
    return float(theta0), info


def estimate_theta_guess_from_strong_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    center_hint_deg: float = 12.8,
    half_window_deg: float = 0.8,
    topk: int = 3,
) -> tuple[float, dict]:
    x = np.asarray(two_theta, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 20:
        return float(center_hint_deg), {"method": "fallback_center_hint_not_enough_points"}

    lo = float(center_hint_deg - half_window_deg)
    hi = float(center_hint_deg + half_window_deg)
    w = (x >= lo) & (x <= hi)
    if np.count_nonzero(w) < 10:
        return float(center_hint_deg), {"method": "fallback_center_hint_not_enough_window_points"}

    xs = x[w]
    ys = y[w]
    ys_s = smooth_1d(ys)
    pks, _ = find_peaks(ys_s, distance=max(1, ys_s.size // 30))
    if pks.size == 0:
        idx = int(np.argmax(ys_s))
        return float(xs[idx]), {"method": "window_argmax"}

    order = np.argsort(ys_s[pks])[::-1]
    kk = max(1, min(int(topk), int(pks.size)))
    top_idx = pks[order[:kk]]
    # From the first few strongest peaks, choose the one closest to center hint.
    # This avoids drifting to unrelated lower-angle strong peaks.
    first_idx = int(top_idx[np.argmin(np.abs(xs[top_idx] - float(center_hint_deg)))])
    theta_guess = float(xs[first_idx])
    info = {
        "method": "closest_to_hint_of_topk_strong_peaks",
        "topk_used": int(kk),
        "theta_candidates_deg": [float(v) for v in np.sort(xs[top_idx])],
    }
    return theta_guess, info


def gaussian_with_offset(x, amp, mu, sigma, offset):
    z = (x - mu) / np.maximum(sigma, 1e-9)
    return amp * np.exp(-0.5 * z * z) + offset


def fit_peak_center_for_row(
    theta: np.ndarray,
    y: np.ndarray,
    theta_guess: float,
    min_row_log10_peak: float = 2.5,
) -> tuple[float, str, float]:
    x = np.asarray(theta, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 8:
        return np.nan, "insufficient_points", np.nan

    row_peak = float(np.nanmax(y))
    row_log10_peak = float(np.log10(max(row_peak, 1e-6)))
    if row_log10_peak < float(min_row_log10_peak):
        return np.nan, "below_log10_threshold", row_log10_peak

    w = (x >= theta_guess - 0.35) & (x <= theta_guess + 0.35)
    if np.count_nonzero(w) >= 8:
        x = x[w]
        y = y[w]

    y_s = smooth_1d(y, max_window=31)
    offset0 = float(np.percentile(y_s, 20))
    amp0 = float(np.max(y_s) - offset0)
    if (not np.isfinite(amp0)) or (amp0 <= 0):
        return np.nan, "no_positive_peak", row_log10_peak

    mu0 = float(x[int(np.argmax(y_s))])
    sigma0 = 0.05

    try:
        popt, _ = curve_fit(
            gaussian_with_offset,
            x,
            y,
            p0=[amp0, mu0, sigma0, offset0],
            bounds=([0.0, float(x.min()), 0.003, -np.inf], [np.inf, float(x.max()), 0.40, np.inf]),
            maxfev=6000,
        )
        return float(popt[1]), "gaussian", row_log10_peak
    except Exception:
        y_pos = np.clip(y - offset0, 0.0, None)
        sw = float(np.sum(y_pos))
        if sw <= 0:
            return np.nan, "fit_failed", row_log10_peak
        mu = float(np.sum(x * y_pos) / sw)
        return mu, "centroid", row_log10_peak


def fit_theta_vs_phi(
    cake: np.ndarray,
    theta_axis: np.ndarray,
    phi_axis: np.ndarray,
    theta_guess: float,
    min_row_log10_peak: float = 2.5,
    max_abs_dev_from_theta0: float | None = 0.25,
    max_abs_dev_from_file_median: float | None = 0.12,
) -> pd.DataFrame:
    cake = np.asarray(cake, dtype=np.float64)
    theta_axis = np.asarray(theta_axis, dtype=np.float64)
    phi_axis = np.asarray(phi_axis, dtype=np.float64)

    rows = []
    for i, phi in enumerate(phi_axis):
        center, method, row_log10_peak = fit_peak_center_for_row(
            theta_axis,
            cake[i, :],
            theta_guess=theta_guess,
            min_row_log10_peak=min_row_log10_peak,
        )
        if np.isfinite(center) and (max_abs_dev_from_theta0 is not None):
            if abs(float(center) - float(theta_guess)) > float(max_abs_dev_from_theta0):
                center = np.nan
                method = "outside_theta_band"
        rows.append(
            {
                "phi_deg": float(phi),
                "theta_center_deg": center,
                "fit_method": method,
                "row_log10_peak": row_log10_peak,
            }
        )
    df = pd.DataFrame(rows)

    # Optional robust cleanup: drop row fits far from the per-file median center.
    # This catches occasional pathological rows (e.g., detector-gap artifacts)
    # that pass peak thresholding but jump to an unrelated center.
    if max_abs_dev_from_file_median is not None and (not df.empty):
        centers = df["theta_center_deg"].to_numpy(dtype=np.float64)
        valid = np.isfinite(centers)
        if int(np.count_nonzero(valid)) >= 8:
            med = float(np.nanmedian(centers[valid]))
            bad = valid & (np.abs(centers - med) > float(max_abs_dev_from_file_median))
            if np.any(bad):
                df.loc[bad, "theta_center_deg"] = np.nan
                df.loc[bad, "fit_method"] = "outside_file_median_band"
    return df


def save_top10_plot(df_top: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    vals = np.asarray(df_top["pt_roi_power_log10"], dtype=np.float64)
    labels = [
        f"{int(r['row'])},{int(r['col'])} | {int(r['frame_index']):03d}"
        for _, r in df_top.iterrows()
    ]
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("log10(pt_roi_power)")
    ax.set_title("Top selected points from Pt map (log ROI power)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_1d_profiles_plot(profile_rows: list[dict], out_png: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in enumerate(profile_rows):
        x = np.asarray(row["two_theta"], dtype=np.float64)
        y = np.asarray(row["intensity"], dtype=np.float64)
        yn = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y) + 1e-12)
        ax.plot(x, yn + i * 1.05, lw=1.0, label=f"{row['frame_index']:03d} theta0={row['theta0']:.4f}")
        ax.plot([row["theta0"]], [np.interp(row["theta0"], x, yn) + i * 1.05], "ro", ms=3)
    ax.set_xlabel("2theta (deg)")
    ax.set_ylabel("normalized I + offset")
    ax.set_title("1D integrations near 12.8 deg and per-file local minima")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_theta_vs_phi_single_plot(df_valid: pd.DataFrame, theta0_file: float, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_valid["phi_deg"], df_valid["theta_center_deg"], ".-", lw=0.8, ms=3)
    ax.axhline(theta0_file, color="tab:red", ls="--", lw=1.2, label=f"theta0(1D)={theta0_file:.4f}")
    ax.set_xlabel("phi (deg)")
    ax.set_ylabel("fitted theta center (deg)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_theta_vs_phi_overlay_plot(df_all: pd.DataFrame, out_png: Path, max_gap_deg: float = 12.0):
    fig, ax = plt.subplots(figsize=(9, 6))
    for file_name, g in df_all.groupby("file"):
        x = np.asarray(g["phi_deg"], dtype=np.float64)
        y = np.asarray(g["theta_center_deg"], dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size == 0:
            continue
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        # Draw points so sparse valid rows are visible without implying continuity.
        ax.scatter(x, y, s=8, alpha=0.35)
        # Break line segments where phi coverage has large gaps.
        split_after = np.where(np.diff(x) > float(max_gap_deg))[0]
        start = 0
        for j in list(split_after) + [x.size - 1]:
            end = int(j) + 1
            if end - start >= 2:
                ax.plot(x[start:end], y[start:end], lw=0.9, alpha=0.45)
            start = end
    ax.set_xlabel("phi (deg)")
    ax.set_ylabel("fitted theta center (deg)")
    ax.set_title(f"Theta center vs phi (all selected files, valid fits only; gap>{max_gap_deg:g} deg broken)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def run(args):
    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (root_dir / "hdf5_images_output" / "poni_theta_phi_check")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[1/6] Output directory: {out_dir}")

    pt_map_path = Path(args.pt_map_path) if args.pt_map_path else (root_dir / "hdf5_images_output" / "maps" / "map_pt_roi_power.npy")
    log(f"[2/6] Loading Pt map: {pt_map_path}")
    df_top = select_top_frames_from_pt_map(
        root_dir,
        pt_map_path,
        top_n=args.top_n,
        skip_first_col=True,
    )
    df_top.to_csv(out_dir / "selected_top_points_from_pt_map.csv", index=False)
    save_top10_plot(df_top, out_dir / "selected_top_points_from_pt_map.png")
    log(f"[2/6] Selected top {len(df_top)} files from log10(pt_roi_power), excluding col=0.")

    log(f"[3/6] Loading PONI: {args.poni}")
    ai = pyFAI.load(args.poni)
    mask_transform = parse_mask_transform(args.mask_transform)

    mask_arr = None
    resolved_mask_path = None

    profile_rows = []
    theta0_rows = []

    log("[3/6] Running 1D integration and theta0 detection per file...")
    for k, (_, r) in enumerate(df_top.iterrows(), start=1):
        file_name = str(r["file"])
        frame_idx = int(frame_index_from_name(file_name))
        path_h5 = root_dir / file_name
        log(f"  [3/6] ({k}/{len(df_top)}) 1D theta0 for {file_name}")
        img = load_image_h5(path_h5, dataset=args.dataset, frame=0, downsample=1)

        if args.mask_source and (mask_arr is None):
            mask_arr, resolved_mask = resolve_mask_for_shape(
                args.mask_source,
                img_shape=img.shape,
                mask_npz_key=args.mask_npz_key,
                mask_transform=mask_transform,
                mask_is_keep_region=bool(args.mask_is_keep_region),
            )
            resolved_mask_path = str(resolved_mask) if resolved_mask is not None else None
            log(f"  [3/6] Resolved mask: {resolved_mask_path}")

        tth1d, I1d = integrate_1d_theta(
            ai,
            img,
            mask=mask_arr,
            npt=args.npt_1d,
            unit=args.unit,
        )
        theta_guess_file = float(args.theta_guess_deg)
        theta_guess_info = {"method": "fixed"}
        if str(args.theta_guess_mode).lower() == "strong_peaks":
            theta_guess_file, theta_guess_info = estimate_theta_guess_from_strong_peaks(
                tth1d,
                I1d,
                center_hint_deg=float(args.theta_guess_deg),
                half_window_deg=float(args.theta_search_half_window_deg),
                topk=int(args.theta_guess_topk_peaks),
            )

        theta0, theta_info = find_theta0_local_minimum(
            tth1d,
            I1d,
            guess_deg=theta_guess_file,
            half_window_deg=args.theta_search_half_window_deg,
        )
        theta0_fallback = False
        if abs(float(theta0) - float(theta_guess_file)) > float(args.theta0_max_deviation_deg):
            peak_theta = theta_info.get("peak_theta_deg", None)
            if peak_theta is not None and np.isfinite(peak_theta):
                theta0 = float(peak_theta)
                theta0_fallback = True
            else:
                theta0 = float(theta_guess_file)
                theta0_fallback = True

        pd.DataFrame({"two_theta_deg": tth1d, "intensity": I1d}).to_csv(
            out_dir / f"profile_1d_{Path(file_name).stem}.csv", index=False
        )

        profile_rows.append(
            {
                "file": file_name,
                "frame_index": frame_idx,
                "two_theta": tth1d,
                "intensity": I1d,
                "theta0": theta0,
            }
        )
        theta0_rows.append(
            {
                "file": file_name,
                "frame_index": frame_idx,
                "row": safe_float(r.get("row", np.nan)),
                "col": safe_float(r.get("col", np.nan)),
                "pt_roi_power": safe_float(r.get("pt_roi_power", np.nan)),
                "pt_roi_power_log10": safe_float(r.get("pt_roi_power_log10", np.nan)),
                "theta0_initial_guess_deg": theta_guess_file,
                "theta0_initial_guess_method": theta_guess_info.get("method", ""),
                "theta0_1d_local_min_deg": theta0,
                "theta0_method": theta_info.get("method", ""),
                "theta0_peak_theta_deg": theta_info.get("peak_theta_deg", None),
                "theta0_left_min_theta_deg": theta_info.get("left_min_theta_deg", None),
                "theta0_right_min_theta_deg": theta_info.get("right_min_theta_deg", None),
                "theta0_fallback_to_peak": bool(theta0_fallback),
            }
        )

    df_theta0 = pd.DataFrame(theta0_rows).sort_values("pt_roi_power_log10", ascending=False).reset_index(drop=True)
    df_theta0.to_csv(out_dir / "theta0_from_1d_top_points.csv", index=False)
    save_1d_profiles_plot(profile_rows, out_dir / "top_points_1d_profiles_theta0.png")
    theta0_by_file = dict(zip(df_theta0["file"], df_theta0["theta0_1d_local_min_deg"]))
    log("[3/6] Per-file theta0 detection complete.")

    all_theta_phi_rows = []
    log("[4/6] Building per-file cake and fitting theta-vs-phi...")
    for k, (_, r) in enumerate(df_top.iterrows(), start=1):
        file_name = str(r["file"])
        frame_idx = int(frame_index_from_name(file_name))
        path_h5 = root_dir / file_name
        log(f"  [4/6] ({k}/{len(df_top)}) cake + fit for {file_name}")
        img = load_image_h5(path_h5, dataset=args.dataset, frame=0, downsample=1)

        theta0_file = float(theta0_by_file[file_name])
        theta_lo = theta0_file - float(args.theta_half_window_deg)
        theta_hi = theta0_file + float(args.theta_half_window_deg)

        cake, tth_axis, phi_axis = integrate_cake_2d(
            img,
            poni_path=args.poni,
            npt_rad=args.npt_rad,
            npt_azim=args.npt_azim,
            unit=args.unit,
            radial_range=(theta_lo, theta_hi),
            mask=mask_arr,
            azimuth_to_0_360=True,
            return_info=False,
        )

        stem = Path(file_name).stem
        cake_png = out_dir / f"cake_{stem}.png"
        cake_npz = out_dir / f"cake_{stem}.npz"
        plot_cake_2d(
            cake,
            tth_axis,
            phi_axis,
            out_png=cake_png,
            log_scale=True,
            title=f"{stem} cake x=2theta y=phi [{theta_lo:.4f},{theta_hi:.4f}]",
        )
        meta = {
            "unit": args.unit,
            "file": file_name,
            "frame_index": frame_idx,
            "theta0_file_deg": theta0_file,
            "theta_range_deg": [theta_lo, theta_hi],
            "mask_source": str(args.mask_source) if args.mask_source else None,
            "resolved_mask_path": resolved_mask_path,
            "mask_transform": mask_transform,
            "mask_is_keep_region": bool(args.mask_is_keep_region),
        }
        save_cake_npz(cake_npz, cake, tth_axis, phi_axis, unit=args.unit, extra_meta=meta)

        df_fp = fit_theta_vs_phi(
            cake,
            tth_axis,
            phi_axis,
            theta_guess=theta0_file,
            min_row_log10_peak=float(args.fit_row_log10_threshold),
            max_abs_dev_from_theta0=float(args.fit_max_abs_dev_from_theta0),
            max_abs_dev_from_file_median=float(args.fit_max_abs_dev_from_file_median),
        )
        df_fp["file"] = file_name
        df_fp["frame_index"] = frame_idx
        df_fp["row"] = safe_float(r.get("row", np.nan))
        df_fp["col"] = safe_float(r.get("col", np.nan))
        df_fp["theta0_file_deg"] = theta0_file
        df_fp["theta_window_lo_deg"] = theta_lo
        df_fp["theta_window_hi_deg"] = theta_hi
        df_fp.to_csv(out_dir / f"theta_vs_phi_raw_{stem}.csv", index=False)

        # Missing fits are allowed: keep only valid points for downstream plots/summary.
        df_valid = df_fp[np.isfinite(df_fp["theta_center_deg"])].copy().reset_index(drop=True)
        df_valid.to_csv(out_dir / f"theta_vs_phi_valid_{stem}.csv", index=False)
        if not df_valid.empty:
            save_theta_vs_phi_single_plot(
                df_valid,
                theta0_file=theta0_file,
                out_png=out_dir / f"theta_vs_phi_{stem}.png",
                title=f"{stem}: theta center vs phi (valid fits)",
            )
            all_theta_phi_rows.append(df_valid)

    if all_theta_phi_rows:
        df_all = pd.concat(all_theta_phi_rows, axis=0, ignore_index=True)
        df_all.to_csv(out_dir / "theta_vs_phi_all_valid.csv", index=False)
        save_theta_vs_phi_overlay_plot(
            df_all,
            out_dir / "theta_vs_phi_overlay_all_valid.png",
            max_gap_deg=float(args.overlay_max_phi_gap_deg),
        )
        valid_count = int(len(df_all))
    else:
        df_all = pd.DataFrame()
        valid_count = 0

    summary = {
        "root_dir": str(root_dir.resolve()),
        "poni_path": str(Path(args.poni).resolve()),
        "dataset": args.dataset,
        "pt_map_path": str(pt_map_path.resolve()),
        "top_n": int(args.top_n),
        "selection_skip_first_col": True,
        "theta_guess_deg": float(args.theta_guess_deg),
        "theta_guess_mode": str(args.theta_guess_mode),
        "theta_guess_topk_peaks": int(args.theta_guess_topk_peaks),
        "theta_search_half_window_deg": float(args.theta_search_half_window_deg),
        "theta_half_window_deg": float(args.theta_half_window_deg),
        "fit_row_log10_threshold": float(args.fit_row_log10_threshold),
        "fit_max_abs_dev_from_theta0": float(args.fit_max_abs_dev_from_theta0),
        "fit_max_abs_dev_from_file_median": float(args.fit_max_abs_dev_from_file_median),
        "theta0_max_deviation_deg": float(args.theta0_max_deviation_deg),
        "overlay_max_phi_gap_deg": float(args.overlay_max_phi_gap_deg),
        "mask_source": str(args.mask_source) if args.mask_source else None,
        "resolved_mask_path": resolved_mask_path,
        "mask_transform": mask_transform,
        "mask_is_keep_region": bool(args.mask_is_keep_region),
        "n_valid_theta_vs_phi_points": valid_count,
        "theta0_per_file": {
            str(r["file"]): float(r["theta0_1d_local_min_deg"]) for _, r in df_theta0.iterrows()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("[5/6] Wrote summary and tables.")
    log("[6/6] Done.")
    print(f"Done. Outputs written to: {out_dir}", flush=True)
    print("Top points selected from log10(pt_roi_power) map:", flush=True)
    print(df_top[["file", "row", "col", "pt_roi_power", "pt_roi_power_log10"]].to_string(index=False), flush=True)
    print("\nPer-file theta0 from 1D local minimum near 12.8 deg:", flush=True)
    print(df_theta0[["file", "theta0_1d_local_min_deg"]].to_string(index=False), flush=True)
    print(f"\nValid theta-vs-phi fitted points saved: {valid_count}", flush=True)


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Select top Pt points from map_pt_roi_power (log scale), find per-file theta0 from 1D local minimum, "
            "then make per-file cake and theta-vs-phi fits."
        )
    )
    p.add_argument(
        "--root-dir",
        type=str,
        default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files",
        help="Directory with restored HDF5 frames.",
    )
    p.add_argument(
        "--pt-map-path",
        type=str,
        default=None,
        help="Path to map_pt_roi_power.npy (default: <root>/hdf5_images_output/maps/map_pt_roi_power.npy).",
    )
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--dataset", type=str, default="data")
    p.add_argument(
        "--poni",
        type=str,
        default=r"E:\XRD\proc\proc\LaB6_003_25keV_poni.poni",
    )
    p.add_argument(
        "--mask-source",
        type=str,
        default=r"F:\NMR\NMR\py_projects\xrd\annulus_phi_mask.mask",
        help="Mask file or directory (optional).",
    )
    p.add_argument("--mask-npz-key", type=str, default=None)
    p.add_argument(
        "--mask-transform",
        type=str,
        default="flipud",
        help="Mask transform: none, flipud, fliplr, transpose, or comma list.",
    )
    p.add_argument(
        "--mask-is-keep-region",
        action="store_true",
        help="If set, mask pixels are interpreted as KEEP region and inverted for pyFAI.",
    )
    p.add_argument("--unit", type=str, default="2th_deg")
    p.add_argument("--npt-1d", type=int, default=5000)
    p.add_argument("--npt-rad", type=int, default=1200)
    p.add_argument("--npt-azim", type=int, default=720)
    p.add_argument("--theta-guess-deg", type=float, default=12.8)
    p.add_argument(
        "--theta-guess-mode",
        type=str,
        default="strong_peaks",
        choices=["fixed", "strong_peaks"],
        help="How to initialize theta guess per file before theta0 detection.",
    )
    p.add_argument(
        "--theta-guess-topk-peaks",
        type=int,
        default=3,
        help="When theta-guess-mode=strong_peaks, use first peak among top-k strongest peaks.",
    )
    p.add_argument("--theta-search-half-window-deg", type=float, default=0.8)
    p.add_argument("--theta-half-window-deg", type=float, default=0.5)
    p.add_argument(
        "--theta0-max-deviation-deg",
        type=float,
        default=0.18,
        help="If per-file theta0 differs from theta-guess by more than this, fallback to local peak theta.",
    )
    p.add_argument(
        "--fit-row-log10-threshold",
        type=float,
        default=2.5,
        help="Only fit phi rows whose row peak intensity has log10 >= threshold.",
    )
    p.add_argument(
        "--fit-max-abs-dev-from-theta0",
        type=float,
        default=0.22,
        help="Discard fitted centers farther than this from per-file theta0.",
    )
    p.add_argument(
        "--fit-max-abs-dev-from-file-median",
        type=float,
        default=0.12,
        help="Discard row fits farther than this from the per-file median fitted center.",
    )
    p.add_argument(
        "--overlay-max-phi-gap-deg",
        type=float,
        default=12.0,
        help="Break overlay line segments when consecutive valid phi points differ by more than this.",
    )
    p.add_argument("--out-dir", type=str, default=None)
    return p


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
