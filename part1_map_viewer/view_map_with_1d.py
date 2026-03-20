from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def map_for_display(metric_name: str, data: np.ndarray) -> tuple[np.ndarray, str]:
    disp = data.copy()
    label = metric_name
    if metric_name in {"spot_power", "ni_score"}:
        finite_pos = disp[np.isfinite(disp) & (disp > 0)]
        if finite_pos.size > 0:
            floor = float(np.percentile(finite_pos, 50))
            disp = np.log10(np.clip(disp, floor, None))
            label = f"log10({metric_name})"
    return disp, label


def load_metric_map(out_dir: Path, metric: str, maps_filename: str) -> np.ndarray:
    map_path = out_dir / "maps" / f"map_{metric}.npy"
    if map_path.exists():
        return np.load(map_path)

    pack_path = out_dir / maps_filename
    if pack_path.exists():
        data = np.load(pack_path)
        if metric in data.files:
            return data[metric]

    raise FileNotFoundError(f"Could not find map '{metric}' in {out_dir}")


def load_map_from_file(map_file: Path) -> tuple[np.ndarray, str]:
    if not map_file.exists():
        raise FileNotFoundError(f"Map file not found: {map_file}")

    suf = map_file.suffix.lower()
    if suf == ".npy":
        return np.load(map_file), map_file.stem

    if suf == ".npz":
        data = np.load(map_file)
        if len(data.files) == 0:
            raise ValueError(f"No arrays in {map_file}")
        key = data.files[0]
        return data[key], f"{map_file.stem}:{key}"

    if suf in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        npy = map_file.with_suffix(".npy")
        if npy.exists():
            print(f"Using sibling raw map instead of image: {npy}")
            return np.load(npy), npy.stem

        img = plt.imread(map_file)
        if img.ndim == 3:
            img = img[..., 0]
        return img.astype(float), map_file.stem

    raise ValueError(f"Unsupported map file format: {map_file}")


def parse_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x_vals: list[float] = []
    y_vals: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 2:
            continue
        x_vals.append(float(cols[0]))
        y_vals.append(float(cols[1]))
    return np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float)


def frame_index_from_name(name: str) -> int:
    m = re.search(r"_(\d+)_restored", name)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else -1


def build_xy_index(sp1d_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in sp1d_dir.glob("*.xy"):
        idx = frame_index_from_name(p.name)
        if idx >= 0:
            out[idx] = p
    if not out:
        raise FileNotFoundError(f"No .xy files found in {sp1d_dir}")
    return out


def load_h5_image(path: Path, dataset_name: str, ds: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        ds0 = f[dataset_name]
        if ds0.ndim == 3 and ds0.shape[0] == 1:
            img = ds0[0, ::ds, ::ds]
        elif ds0.ndim == 2:
            img = ds0[::ds, ::ds]
        else:
            raise ValueError(f"Unsupported dataset shape {ds0.shape} in {path.name}")
    return img.astype(np.float32)


def launch_viewer(
    root_dir: Path,
    out_dir: Path,
    sp1d_dir: Path,
    metric: str,
    metrics_filename: str,
    maps_filename: str,
    map_file: Path | None,
    ny: int,
    nx: int,
    col_shift: int,
    yscale: str,
    x_min: float | None,
    x_max: float | None,
    right_mode: str,
    dataset_name: str,
    show_downsample: int,
    flip_ud: bool,
    flip_lr: bool,
    flip_right_ud: bool,
    flip_right_lr: bool,
) -> None:
    df = pd.read_csv(out_dir / metrics_filename)
    if map_file is not None:
        metric_map, map_name = load_map_from_file(map_file)
        map_disp, map_label = map_for_display(map_name, metric_map)
    else:
        metric_map = load_metric_map(out_dir, metric, maps_filename=maps_filename)
        map_disp, map_label = map_for_display(metric, metric_map)

    map_show = map_disp.copy()
    if flip_ud:
        map_show = np.flipud(map_show)
    if flip_lr:
        map_show = np.fliplr(map_show)

    map_ny, map_nx = int(map_show.shape[0]), int(map_show.shape[1])
    if (map_ny, map_nx) != (ny, nx):
        print(f"Using map shape for click bounds: ({map_ny}, {map_nx}) instead of ({ny}, {nx})")

    xy_by_index = build_xy_index(sp1d_dir) if right_mode == "spectrum" else {}

    frame_lookup: dict[tuple[int, int], int] = {}
    file_lookup: dict[tuple[int, int], str] = {}
    for _, r in df.iterrows():
        row = int(r["row"])
        col = int(r["col"])
        fname = str(r["file"])
        frame_lookup[(row, col)] = frame_index_from_name(fname)
        file_lookup[(row, col)] = fname

    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    ax_map = fig.add_subplot(1, 2, 1)
    ax_right = fig.add_subplot(1, 2, 2)

    im_map = ax_map.imshow(map_show, origin="lower", aspect="auto")
    cbar = fig.colorbar(im_map, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label(map_label)
    ax_map.set_title(f"Click map: {map_label}")
    ax_map.set_xlabel("col")
    ax_map.set_ylabel("row")

    marker = {"artist": None}
    state = {"row": None, "col": None, "row_plot": None, "col_plot": None, "yscale": yscale}

    def redraw_right(row: int, col: int, row_plot: int, col_plot: int) -> None:
        key = (row, col)
        if key not in file_lookup:
            ax_right.clear()
            ax_right.set_title(f"No mapping for row={row}, col={col}")
            ax_right.set_axis_off()
            fig.canvas.draw_idle()
            return

        h5_path = root_dir / file_lookup[key]

        if right_mode == "image":
            try:
                img = load_h5_image(h5_path, dataset_name=dataset_name, ds=show_downsample)
            except Exception as e:
                ax_right.clear()
                ax_right.set_title(f"Failed loading image:\n{e}")
                ax_right.set_axis_off()
                fig.canvas.draw_idle()
                return

            finite = np.isfinite(img)
            vals = img[finite]
            if vals.size == 0:
                disp = np.zeros_like(img)
                vmin, vmax = 0.0, 1.0
            else:
                floor = max(float(np.percentile(vals, 1)), 1e-6)
                disp = np.log10(np.clip(img, floor, None))
                vmin = float(np.nanpercentile(disp, 75))
                vmax = float(np.nanpercentile(disp, 99.7))

            if flip_right_ud:
                disp = np.flipud(disp)
            if flip_right_lr:
                disp = np.fliplr(disp)

            ax_right.clear()
            ax_right.imshow(disp, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
            ax_right.set_title(f"{h5_path.name} | row={row}, col={col}, ds={show_downsample}")
            ax_right.axis("off")

        else:
            frame_idx = frame_lookup.get(key, -1)
            sp_path = xy_by_index.get(frame_idx)

            if sp_path is None:
                ax_right.clear()
                ax_right.set_title(f"No .xy for frame index {frame_idx}")
                ax_right.set_axis_off()
                fig.canvas.draw_idle()
                return

            x, y = parse_xy(sp_path)
            ax_right.clear()
            ax_right.plot(x, y, lw=1.0, color="tab:blue")
            ax_right.set_yscale(state["yscale"])
            ax_right.set_xlabel(r"2$\\theta$ (deg)")
            ax_right.set_ylabel("Intensity")

            peak_pos = np.nan
            if 0 <= row < metric_map.shape[0] and 0 <= col < metric_map.shape[1]:
                peak_pos = float(metric_map[row, col])
            if np.isfinite(peak_pos):
                ax_right.axvline(peak_pos, ls="--", lw=1.3, color="darkorange", alpha=0.9)

            title = f"{sp_path.name}\\nframe={frame_idx} | row={row}, col={col} | y={state['yscale']}"
            if np.isfinite(peak_pos):
                title += f" | peak={peak_pos:.4f}"
            ax_right.set_title(title)

            if x_min is not None or x_max is not None:
                ax_right.set_xlim(left=x_min, right=x_max)
            ax_right.grid(alpha=0.2)

        if marker["artist"] is not None:
            marker["artist"].remove()
        marker["artist"] = ax_map.plot(
            col_plot,
            row_plot,
            marker="s",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor="cyan",
            markeredgewidth=2,
        )[0]
        fig.canvas.draw_idle()

    def onclick(event) -> None:
        if event.inaxes is not ax_map:
            return
        if event.xdata is None or event.ydata is None:
            return

        col_plot = int(np.round(event.xdata))
        row_plot = int(np.round(event.ydata))
        if row_plot < 0 or row_plot >= map_ny or col_plot < 0 or col_plot >= map_nx:
            return

        row = (map_ny - 1 - row_plot) if flip_ud else row_plot
        col_base = (map_nx - 1 - col_plot) if flip_lr else col_plot
        col = (col_base + col_shift) % map_nx

        state["row"] = row
        state["col"] = col
        state["row_plot"] = row_plot
        state["col_plot"] = col_plot
        redraw_right(row, col, row_plot, col_plot)

    def onkey(event) -> None:
        if right_mode == "spectrum" and event.key in ("l", "L"):
            state["yscale"] = "linear" if state["yscale"] == "log" else "log"
            if (
                state["row"] is not None
                and state["col"] is not None
                and state["row_plot"] is not None
                and state["col_plot"] is not None
            ):
                redraw_right(
                    int(state["row"]),
                    int(state["col"]),
                    int(state["row_plot"]),
                    int(state["col_plot"]),
                )

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)

    plt.tight_layout()
    plt.show(block=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive map -> right-panel viewer (spectrum or image).")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(r"E:\XRD\data\PIMEGA_gridscan_00_47_31_restored_files"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--sp1d-dir",
        type=Path,
        default=None,
        help="Directory with integrated .xy files (default: <out-dir>/linecut/allsp)",
    )
    parser.add_argument("--metric", type=str, default="ni_score")
    parser.add_argument(
        "--map-file",
        type=Path,
        default=None,
        help="Custom map file (.npy/.npz or image; image auto-uses sibling .npy if available)",
    )
    parser.add_argument("--metrics-filename", type=str, default="xrd_metrics_ni.csv")
    parser.add_argument("--maps-filename", type=str, default="xrd_maps_ni.npz")
    parser.add_argument("--ny", type=int, default=31)
    parser.add_argument("--nx", type=int, default=33)
    parser.add_argument(
        "--col-shift",
        type=int,
        default=0,
        help="display_col -> data_col transform: data_col=(display_col + col_shift) %% nx",
    )
    parser.add_argument("--yscale", choices=["linear", "log"], default="linear")
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--right-mode", choices=["spectrum", "image"], default="spectrum")
    parser.add_argument("--dataset-name", type=str, default="data")
    parser.add_argument("--show-downsample", type=int, default=1)
    parser.add_argument("--flip-ud", action="store_true", help="Flip left map vertically to match other plots")
    parser.add_argument("--flip-lr", action="store_true", help="Flip left map horizontally to match other plots")
    parser.add_argument("--flip-right-ud", action="store_true", help="Flip right image vertically")
    parser.add_argument("--flip-right-lr", action="store_true", help="Flip right image horizontally")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.root_dir / "hdf5_images_output")
    sp1d_dir = args.sp1d_dir or (out_dir / "linecut" / "allsp")

    launch_viewer(
        root_dir=args.root_dir,
        out_dir=out_dir,
        sp1d_dir=sp1d_dir,
        metric=args.metric,
        metrics_filename=args.metrics_filename,
        maps_filename=args.maps_filename,
        map_file=args.map_file,
        ny=args.ny,
        nx=args.nx,
        col_shift=args.col_shift,
        yscale=args.yscale,
        x_min=args.x_min,
        x_max=args.x_max,
        right_mode=args.right_mode,
        dataset_name=args.dataset_name,
        show_downsample=args.show_downsample,
        flip_ud=args.flip_ud,
        flip_lr=args.flip_lr,
        flip_right_ud=args.flip_right_ud,
        flip_right_lr=args.flip_right_lr,
    )


if __name__ == "__main__":
    main()

