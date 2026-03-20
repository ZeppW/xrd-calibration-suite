from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def sine_model(phi_deg: np.ndarray, offset: float, amplitude: float, phase_deg: float) -> np.ndarray:
    # Period is fixed at 360 deg in phi.
    return offset + amplitude * np.sin(np.deg2rad(phi_deg - phase_deg))


def run(args):
    csv_path = Path(args.input_csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {"file", "phi_deg", "theta_center_deg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    df = df[np.isfinite(df["phi_deg"]) & np.isfinite(df["theta_center_deg"])].copy()
    if df.empty:
        raise RuntimeError("No valid rows after filtering finite phi/theta values.")

    # Normalize each file's curve by its own average theta center.
    means = df.groupby("file")["theta_center_deg"].mean().rename("theta_mean_file")
    df = df.join(means, on="file")
    df["theta_norm"] = df["theta_center_deg"] / df["theta_mean_file"]
    df["theta_norm_delta"] = df["theta_norm"] - 1.0
    df.to_csv(out_dir / "theta_vs_phi_normalized_all.csv", index=False)

    # Joint curve (sum/mean by phi). Mean is sum divided by count.
    joint = (
        df.groupby("phi_deg", as_index=False)
        .agg(
            theta_norm_mean=("theta_norm", "mean"),
            theta_norm_std=("theta_norm", "std"),
            theta_norm_sum=("theta_norm", "sum"),
            n=("theta_norm", "size"),
        )
        .sort_values("phi_deg")
        .reset_index(drop=True)
    )
    joint.to_csv(out_dir / "theta_vs_phi_joint_normalized.csv", index=False)

    fit_df = joint[joint["n"] >= int(args.min_count)].copy()
    if len(fit_df) < 6:
        raise RuntimeError(
            f"Not enough phi points for sine fit after min_count={args.min_count}. "
            f"Have {len(fit_df)} points."
        )

    x = fit_df["phi_deg"].to_numpy(dtype=np.float64)
    y = fit_df["theta_norm_mean"].to_numpy(dtype=np.float64)
    w = fit_df["n"].to_numpy(dtype=np.float64)

    p0 = [float(np.mean(y)), float((np.max(y) - np.min(y)) * 0.5), 0.0]
    sigma = 1.0 / np.sqrt(np.maximum(w, 1.0))
    popt, pcov = curve_fit(
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

    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan, np.nan, np.nan]
    summary = {
        "input_csv": str(csv_path.resolve()),
        "n_rows_valid": int(len(df)),
        "n_files": int(df["file"].nunique()),
        "n_joint_points": int(len(joint)),
        "n_fit_points_after_min_count": int(len(fit_df)),
        "fit_min_count": int(args.min_count),
        "offset": float(popt[0]),
        "amplitude": float(popt[1]),
        "phase_deg": float(popt[2]),
        "offset_stderr": float(perr[0]),
        "amplitude_stderr": float(perr[1]),
        "phase_deg_stderr": float(perr[2]),
        "r2": r2,
        "rmse": rmse,
    }
    (out_dir / "theta_vs_phi_joint_sine_fit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        joint["phi_deg"],
        joint["theta_norm_mean"],
        s=np.clip(joint["n"].to_numpy(dtype=float) * 3.0, 10.0, 60.0),
        alpha=0.6,
        label="joint normalized mean",
    )
    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 1200)
    y_dense = sine_model(x_dense, *popt)
    ax.plot(x_dense, y_dense, "r-", lw=2.0, label="sine fit")
    ax.set_xlabel("phi (deg)")
    ax.set_ylabel("normalized theta (theta / mean_file)")
    ax.set_title(
        "Joint normalized curve and sine fit\n"
        f"offset={popt[0]:.6f}, amp={popt[1]:.6f}, phase={popt[2]:.2f} deg, R^2={r2:.4f}"
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "theta_vs_phi_joint_sine_fit.png", dpi=180)
    plt.close(fig)

    print(f"Wrote outputs to: {out_dir}")
    print(json.dumps(summary, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description="Normalize per-file theta-vs-phi curves and fit joint sine curve.")
    p.add_argument(
        "--input-csv",
        type=str,
        default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files\hdf5_images_output\poni_theta_phi_check\theta_vs_phi_all_valid.csv",
    )
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Require at least this many contributing files per phi for fitting.",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
