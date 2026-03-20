from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def sine_model(phi_deg: np.ndarray, offset: float, amplitude: float, phase_deg: float) -> np.ndarray:
    return offset + amplitude * np.sin(np.deg2rad(phi_deg - phase_deg))


def parse_file_key(name: str) -> str:
    stem = Path(name).stem
    prefix = "theta_vs_phi_valid_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def frame_idx_from_key(key: str) -> int:
    m = re.search(r"_(\d+)_restored", key)
    return int(m.group(1)) if m else -1


def load_centered_merged(folder: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.glob("theta_vs_phi_valid_*.csv")):
        key = parse_file_key(p.name)
        frame_idx = frame_idx_from_key(key)
        df = pd.read_csv(p)
        if not {"phi_deg", "theta_center_deg"} <= set(df.columns):
            continue
        df = df[np.isfinite(df["phi_deg"]) & np.isfinite(df["theta_center_deg"])].copy()
        if df.empty:
            continue
        theta_mean = float(df["theta_center_deg"].mean())
        df["theta_mean_file"] = theta_mean
        df["theta_centered_deg"] = df["theta_center_deg"] - theta_mean
        df["key"] = key
        df["frame_index"] = frame_idx
        rows.append(df[["key", "frame_index", "phi_deg", "theta_center_deg", "theta_mean_file", "theta_centered_deg"]])
    if not rows:
        return pd.DataFrame(columns=["key", "frame_index", "phi_deg", "theta_center_deg", "theta_mean_file", "theta_centered_deg"])
    out = pd.concat(rows, axis=0, ignore_index=True)
    out = out[np.isfinite(out["phi_deg"]) & np.isfinite(out["theta_centered_deg"])].copy()
    return out


def fit_merged_sine(df: pd.DataFrame) -> dict:
    x = df["phi_deg"].to_numpy(dtype=np.float64)
    y = df["theta_centered_deg"].to_numpy(dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 10:
        return {
            "ok": False,
            "n_points": int(x.size),
            "offset": np.nan,
            "amplitude": np.nan,
            "phase_deg": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
        }

    p0 = [float(np.mean(y)), float(0.5 * (np.percentile(y, 95) - np.percentile(y, 5))), 0.0]
    popt, _ = curve_fit(sine_model, x, y, p0=p0, maxfev=40000)
    y_fit = sine_model(x, *popt)
    resid = y - y_fit
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid * resid)))
    return {
        "ok": True,
        "n_points": int(x.size),
        "offset": float(popt[0]),
        "amplitude": float(popt[1]),
        "phase_deg": float(popt[2]),
        "r2": r2,
        "rmse": rmse,
    }


def phi_binned_mean(df: pd.DataFrame, bin_deg: float = 2.0) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["phi_bin_center_deg", "theta_centered_mean_deg", "n"])
    bb = np.floor(df["phi_deg"] / float(bin_deg)).astype(int)
    tmp = df.copy()
    tmp["phi_bin"] = bb
    g = tmp.groupby("phi_bin", as_index=False).agg(
        theta_centered_mean_deg=("theta_centered_deg", "mean"),
        n=("theta_centered_deg", "size"),
    )
    g["phi_bin_center_deg"] = (g["phi_bin"].to_numpy(dtype=float) + 0.5) * float(bin_deg)
    return g[["phi_bin_center_deg", "theta_centered_mean_deg", "n"]].sort_values("phi_bin_center_deg")


def plot_comparison(df0: pd.DataFrame, fit0: dict, df1: pd.DataFrame, fit1: dict, out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    panels = [("Original PONI", df0, fit0), ("Best PONI", df1, fit1)]
    for ax, (title, df, fit) in zip(axes, panels):
        ax.scatter(df["phi_deg"], df["theta_centered_deg"], s=7, alpha=0.2)
        b = phi_binned_mean(df, bin_deg=2.0)
        if not b.empty:
            ax.plot(b["phi_bin_center_deg"], b["theta_centered_mean_deg"], "k.", ms=3, alpha=0.7, label="2deg-bin mean")
        if fit.get("ok", False):
            xg = np.linspace(float(df["phi_deg"].min()), float(df["phi_deg"].max()), 1200)
            yg = sine_model(xg, fit["offset"], fit["amplitude"], fit["phase_deg"])
            ax.plot(
                xg,
                yg,
                "r-",
                lw=2.0,
                label=f"sine: amp={fit['amplitude']:.3e}, phase={fit['phase_deg']:.2f}, R2={fit['r2']:.4f}",
            )
        ax.set_title(title)
        ax.set_xlabel("phi (deg)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("theta_centered = theta - mean_file (deg)")
    fig.suptitle("Merged centered points and sine fit")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def run(args):
    dir_original = Path(args.dir_original)
    dir_best = Path(args.dir_best)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_orig = load_centered_merged(dir_original)
    df_best = load_centered_merged(dir_best)

    if df_orig.empty or df_best.empty:
        raise RuntimeError("Missing valid theta_vs_phi_valid data in original/best directories.")

    df_orig.to_csv(out_dir / "merged_centered_original.csv", index=False)
    df_best.to_csv(out_dir / "merged_centered_best.csv", index=False)

    fit_orig = fit_merged_sine(df_orig)
    fit_best = fit_merged_sine(df_best)

    summary = {
        "original": fit_orig,
        "best": fit_best,
        "abs_amp_improvement": float(abs(fit_orig.get("amplitude", np.nan)) - abs(fit_best.get("amplitude", np.nan))),
        "n_points_original": int(len(df_orig)),
        "n_points_best": int(len(df_best)),
    }
    (out_dir / "merged_centered_sine_fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "label": "original",
                **fit_orig,
            },
            {
                "label": "best",
                **fit_best,
            },
        ]
    ).to_csv(out_dir / "merged_centered_sine_fit_summary.csv", index=False)

    plot_comparison(df_orig, fit_orig, df_best, fit_best, out_dir / "merged_centered_sine_fit_original_vs_best.png")

    print(f"Wrote outputs to: {out_dir}")
    print(json.dumps(summary, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description="Subtract per-file mean, merge points, then sine-fit the merged cloud.")
    p.add_argument(
        "--dir-original",
        type=str,
        default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files\hdf5_images_output\poni_compare\original",
    )
    p.add_argument(
        "--dir-best",
        type=str,
        default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files\hdf5_images_output\poni_compare\best",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files\hdf5_images_output\poni_compare\comparison_plots",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
