from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyFAI

from poni_theta_phi_check import build_parser as build_theta_parser
from poni_theta_phi_check import run as run_theta_phi_check
from theta_phi_joint_sine_fit import build_parser as build_sine_parser
from theta_phi_joint_sine_fit import run as run_theta_phi_joint_sine_fit


def log(msg: str):
    print(msg, flush=True)


def format_tag(dx: float, dy: float, dtilt: float, dtiltplan: float, label: str | None) -> str:
    base = f"dx{dx:+.3f}_dy{dy:+.3f}_dtilt{dtilt:+.3f}_dtiltplan{dtiltplan:+.3f}"
    if label:
        return f"{label}_{base}"
    return base


def write_candidate_poni(base_poni: Path, out_poni: Path, dx: float, dy: float, dtilt: float, dtiltplan: float):
    ai = pyFAI.load(str(base_poni))
    fit2d = ai.getFit2D()

    new_center_x = float(fit2d["centerX"]) + float(dx)
    new_center_y = float(fit2d["centerY"]) + float(dy)
    new_tilt = float(fit2d["tilt"]) + float(dtilt)
    new_tilt_plan = float(fit2d["tiltPlanRotation"]) + float(dtiltplan)

    ai.setFit2D(
        directDist=float(fit2d["directDist"]),
        centerX=new_center_x,
        centerY=new_center_y,
        tilt=new_tilt,
        tiltPlanRotation=new_tilt_plan,
        pixelX=float(fit2d["pixelX"]),
        pixelY=float(fit2d["pixelY"]),
        splinefile=fit2d.get("splinefile", None),
        wavelength=fit2d.get("wavelength", None),
    )
    out_poni.parent.mkdir(parents=True, exist_ok=True)
    ai.save(str(out_poni))

    out_fit = ai.getFit2D()
    return fit2d, out_fit


def append_history_csv(history_csv: Path, row: dict):
    import csv

    history_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = history_csv.exists()
    fields = [
        "tag",
        "candidate_poni",
        "dx",
        "dy",
        "dtilt",
        "dtiltplan",
        "centerX",
        "centerY",
        "tilt",
        "tiltPlanRotation",
        "sine_amplitude",
        "sine_phase_deg",
        "sine_offset",
        "sine_r2",
        "sine_rmse",
        "n_fit_points",
    ]
    with history_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def run(args):
    base_poni = Path(args.base_poni)
    out_root = Path(args.out_root)
    tag = format_tag(args.dx, args.dy, args.dtilt, args.dtiltplan, args.label)
    trial_dir = out_root / tag
    candidate_poni = trial_dir / "candidate.poni"

    log(f"[1/4] Creating candidate PONI: {candidate_poni}")
    _, new_fit2d = write_candidate_poni(
        base_poni=base_poni,
        out_poni=candidate_poni,
        dx=float(args.dx),
        dy=float(args.dy),
        dtilt=float(args.dtilt),
        dtiltplan=float(args.dtiltplan),
    )
    log(
        "[1/4] New Fit2D: "
        f"centerX={new_fit2d['centerX']:.4f}, centerY={new_fit2d['centerY']:.4f}, "
        f"tilt={new_fit2d['tilt']:.6f}, tiltPlan={new_fit2d['tiltPlanRotation']:.6f}"
    )

    log("[2/4] Running theta-phi check pipeline...")
    theta_parser = build_theta_parser()
    theta_args = theta_parser.parse_args(
        [
            "--root-dir",
            str(Path(args.root_dir)),
            "--poni",
            str(candidate_poni),
            "--mask-source",
            str(Path(args.mask_source)),
            "--top-n",
            str(int(args.top_n)),
            "--fit-row-log10-threshold",
            str(float(args.fit_row_log10_threshold)),
            "--fit-max-abs-dev-from-theta0",
            str(float(args.fit_max_abs_dev_from_theta0)),
            "--fit-max-abs-dev-from-file-median",
            str(float(args.fit_max_abs_dev_from_file_median)),
            "--theta-search-half-window-deg",
            str(float(args.theta_search_half_window_deg)),
            "--theta-half-window-deg",
            str(float(args.theta_half_window_deg)),
            "--out-dir",
            str(trial_dir),
        ]
    )
    run_theta_phi_check(theta_args)

    input_csv = trial_dir / "theta_vs_phi_all_valid.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing pipeline output CSV: {input_csv}")

    log("[3/4] Fitting joint sine curve on normalized data...")
    sine_parser = build_sine_parser()
    sine_args = sine_parser.parse_args(
        [
            "--input-csv",
            str(input_csv),
            "--out-dir",
            str(trial_dir),
            "--min-count",
            str(int(args.sine_min_count)),
        ]
    )
    run_theta_phi_joint_sine_fit(sine_args)

    fit_json = trial_dir / "theta_vs_phi_joint_sine_fit.json"
    summary = json.loads(fit_json.read_text(encoding="utf-8"))

    row = {
        "tag": tag,
        "candidate_poni": str(candidate_poni),
        "dx": float(args.dx),
        "dy": float(args.dy),
        "dtilt": float(args.dtilt),
        "dtiltplan": float(args.dtiltplan),
        "centerX": float(new_fit2d["centerX"]),
        "centerY": float(new_fit2d["centerY"]),
        "tilt": float(new_fit2d["tilt"]),
        "tiltPlanRotation": float(new_fit2d["tiltPlanRotation"]),
        "sine_amplitude": float(summary["amplitude"]),
        "sine_phase_deg": float(summary["phase_deg"]),
        "sine_offset": float(summary["offset"]),
        "sine_r2": float(summary["r2"]),
        "sine_rmse": float(summary["rmse"]),
        "n_fit_points": int(summary["n_fit_points_after_min_count"]),
    }
    append_history_csv(Path(args.history_csv), row)

    log("[4/4] Done.")
    print("\nTrial summary:", flush=True)
    print(json.dumps(row, indent=2), flush=True)
    print(f"\nArtifacts: {trial_dir}", flush=True)


def build_parser():
    p = argparse.ArgumentParser(
        description="Adjust PONI Fit2D (centerX/centerY/tilt), rerun theta-phi workflow, and report sine-fit amplitude."
    )
    p.add_argument("--base-poni", type=str, default=r"E:\XRD\proc\proc\LaB6_003_25keV_poni.poni")
    p.add_argument("--root-dir", type=str, default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files")
    p.add_argument("--mask-source", type=str, default=r"F:\NMR\NMR\py_projects\xrd\annulus_phi_mask.mask")

    p.add_argument("--dx", type=float, default=0.0, help="Delta centerX in pixels.")
    p.add_argument("--dy", type=float, default=0.0, help="Delta centerY in pixels.")
    p.add_argument("--dtilt", type=float, default=0.0, help="Delta tilt in degrees.")
    p.add_argument("--dtiltplan", type=float, default=0.0, help="Delta tiltPlanRotation in degrees.")
    p.add_argument("--label", type=str, default="", help="Optional trial label prefix.")

    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--fit-row-log10-threshold", type=float, default=2.5)
    p.add_argument("--fit-max-abs-dev-from-theta0", type=float, default=0.2)
    p.add_argument("--fit-max-abs-dev-from-file-median", type=float, default=0.12)
    p.add_argument("--theta-search-half-window-deg", type=float, default=0.8)
    p.add_argument("--theta-half-window-deg", type=float, default=0.5)
    p.add_argument("--sine-min-count", type=int, default=2)

    p.add_argument(
        "--out-root",
        type=str,
        default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files\hdf5_images_output\poni_tuning_trials",
    )
    p.add_argument(
        "--history-csv",
        type=str,
        default=r"D:\xrd\PIMEGA_gridscan_17_54_21_restored_files\hdf5_images_output\poni_tuning_trials\tuning_history.csv",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
