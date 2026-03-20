from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def sine_model(phi_deg: np.ndarray, offset: float, amplitude: float, phase_deg: float) -> np.ndarray:
    return offset + amplitude * np.sin(np.deg2rad(phi_deg - phase_deg))


def parse_file_key(name: str) -> str:
    # from theta_vs_phi_valid_PIMEGA..._restored.csv -> PIMEGA..._restored
    stem = Path(name).stem
    prefix = "theta_vs_phi_valid_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def frame_idx_from_key(key: str) -> int:
    m = re.search(r"_(\d+)_restored", key)
    return int(m.group(1)) if m else -1


def fit_sine(phi_deg: np.ndarray, theta_deg: np.ndarray) -> dict:
    x = np.asarray(phi_deg, dtype=np.float64)
    y = np.asarray(theta_deg, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 8:
        return {
            "ok": False,
            "n": int(x.size),
            "offset": np.nan,
            "amplitude": np.nan,
            "phase_deg": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
        }

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    p0 = [float(np.mean(y)), float(0.5 * (np.max(y) - np.min(y))), 0.0]
    try:
        popt, _ = curve_fit(sine_model, x, y, p0=p0, maxfev=20000)
    except Exception:
        return {
            "ok": False,
            "n": int(x.size),
            "offset": np.nan,
            "amplitude": np.nan,
            "phase_deg": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
        }

    y_fit = sine_model(x, *popt)
    resid = y - y_fit
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid * resid)))
    return {
        "ok": True,
        "n": int(x.size),
        "offset": float(popt[0]),
        "amplitude": float(popt[1]),
        "phase_deg": float(popt[2]),
        "r2": r2,
        "rmse": rmse,
    }


def load_valid_curves(folder: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(folder.glob("theta_vs_phi_valid_*.csv")):
        key = parse_file_key(p.name)
        df = pd.read_csv(p)
        if {"phi_deg", "theta_center_deg"} <= set(df.columns):
            df = df[np.isfinite(df["phi_deg"]) & np.isfinite(df["theta_center_deg"])].copy()
            out[key] = df
    return out


def run(args):
    dir_orig = Path(args.dir_original)
    dir_best = Path(args.dir_best)
    out_dir = Path(args.out_dir) if args.out_dir else dir_best.parent / "poni_compare_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    curves_orig = load_valid_curves(dir_orig)
    curves_best = load_valid_curves(dir_best)

    keys = sorted(set(curves_orig.keys()) & set(curves_best.keys()), key=frame_idx_from_key)
    if not keys:
        raise RuntimeError("No common theta_vs_phi_valid files found between original and best folders.")

    rows = []
    n = len(keys)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(10, nrows * 3.4)), squeeze=False)

    for i, key in enumerate(keys):
        ax = axes[i // ncols, i % ncols]
        d0 = curves_orig[key]
        d1 = curves_best[key]

        f0 = fit_sine(d0["phi_deg"].to_numpy(), d0["theta_center_deg"].to_numpy())
        f1 = fit_sine(d1["phi_deg"].to_numpy(), d1["theta_center_deg"].to_numpy())

        rows.append(
            {
                "key": key,
                "frame_index": frame_idx_from_key(key),
                "n_orig": f0["n"],
                "amp_orig": f0["amplitude"],
                "phase_orig_deg": f0["phase_deg"],
                "r2_orig": f0["r2"],
                "rmse_orig": f0["rmse"],
                "n_best": f1["n"],
                "amp_best": f1["amplitude"],
                "phase_best_deg": f1["phase_deg"],
                "r2_best": f1["r2"],
                "rmse_best": f1["rmse"],
                "abs_amp_improvement": (abs(f0["amplitude"]) - abs(f1["amplitude"])),
            }
        )

        ax.scatter(d0["phi_deg"], d0["theta_center_deg"], s=8, alpha=0.35, label="orig data", color="tab:blue")
        ax.scatter(d1["phi_deg"], d1["theta_center_deg"], s=8, alpha=0.35, label="best data", color="tab:orange")

        xx0 = np.linspace(float(d0["phi_deg"].min()), float(d0["phi_deg"].max()), 500)
        xx1 = np.linspace(float(d1["phi_deg"].min()), float(d1["phi_deg"].max()), 500)
        if f0["ok"]:
            ax.plot(xx0, sine_model(xx0, f0["offset"], f0["amplitude"], f0["phase_deg"]), color="tab:blue", lw=2.0)
        if f1["ok"]:
            ax.plot(xx1, sine_model(xx1, f1["offset"], f1["amplitude"], f1["phase_deg"]), color="tab:orange", lw=2.0)

        ax.set_title(
            f"{key}\n|amp| orig={abs(f0['amplitude']):.2e}, best={abs(f1['amplitude']):.2e}"
        )
        ax.set_xlabel("phi (deg)")
        ax.set_ylabel("theta center (deg)")
        ax.grid(alpha=0.2)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Per-point sine fit comparison: original PONI vs best PONI", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_dir / "sine_fit_per_point_original_vs_best.png", dpi=180)
    plt.close(fig)

    df_sum = pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)
    df_sum.to_csv(out_dir / "sine_fit_per_point_summary.csv", index=False)

    # Bar chart of |amp| before/after for quick inspection.
    x = np.arange(len(df_sum))
    width = 0.38
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.bar(x - width / 2, np.abs(df_sum["amp_orig"].to_numpy()), width=width, label="|amp| original", color="tab:blue")
    ax2.bar(x + width / 2, np.abs(df_sum["amp_best"].to_numpy()), width=width, label="|amp| best", color="tab:orange")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(int(v)) for v in df_sum["frame_index"].to_numpy()], rotation=45, ha="right")
    ax2.set_xlabel("frame index")
    ax2.set_ylabel("|sine amplitude| (deg)")
    ax2.set_title("Per-point sine amplitude: original vs best PONI")
    ax2.grid(axis="y", alpha=0.2)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "sine_amp_per_point_original_vs_best.png", dpi=180)
    plt.close(fig2)

    print(f"Wrote: {out_dir}")
    print(f"Points compared: {len(df_sum)}")
    print(df_sum[["frame_index", "amp_orig", "amp_best", "abs_amp_improvement"]].to_string(index=False))


def build_parser():
    p = argparse.ArgumentParser(description="Compare per-point sine fit (theta vs phi) for original vs best PONI.")
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
