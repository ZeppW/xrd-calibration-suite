from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def frame_index_from_name(name: str) -> int:
    m = re.search(r"_(\d+)_restored", name)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else -1


def load_map_npy(map_npy: Path) -> np.ndarray:
    if not map_npy.exists():
        raise FileNotFoundError(f"Map NPY not found: {map_npy}")
    if map_npy.suffix.lower() != ".npy":
        raise ValueError(f"Map must be .npy: {map_npy}")

    data = np.load(map_npy)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Map .npy must be 2D; got shape={data.shape}")
    if not np.any(np.isfinite(data)):
        raise ValueError(f"Map .npy has no finite values: {map_npy}")
    return data


def map_for_display(data: np.ndarray) -> tuple[np.ndarray, str]:
    disp = data.astype(np.float32).copy()
    finite_pos = disp[np.isfinite(disp) & (disp > 0)]
    label = "map value"
    if finite_pos.size > 0:
        floor = max(float(np.percentile(finite_pos, 50)), 1e-8)
        disp = np.log10(np.clip(disp, floor, None))
        label = "log10(map value)"
    return disp, label


def sorted_h5_files(root_dir: Path) -> tuple[dict[int, Path], list[Path]]:
    files = sorted(root_dir.glob("*_restored.hdf5"), key=lambda p: frame_index_from_name(p.name))
    if not files:
        raise FileNotFoundError(f"No '*_restored.hdf5' files found in {root_dir}")

    by_index: dict[int, Path] = {}
    for p in files:
        idx = frame_index_from_name(p.name)
        if idx >= 1:
            by_index[idx] = p

    return by_index, files


def _dataset_is_image_like(handle: h5py.File, dataset_name: str) -> bool:
    if dataset_name not in handle:
        return False
    ds = handle[dataset_name]
    return isinstance(ds, h5py.Dataset) and ds.ndim >= 2 and np.issubdtype(ds.dtype, np.number)


def choose_image_dataset(handle: h5py.File) -> str:
    if _dataset_is_image_like(handle, "data"):
        return "data"

    candidates: list[tuple[str, int, int]] = []

    def visitor(name: str, obj: object) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if obj.ndim < 2:
            return
        if not np.issubdtype(obj.dtype, np.number):
            return
        is_2d = 1 if obj.ndim == 2 else 0
        plane_size = int(obj.shape[-2] * obj.shape[-1])
        candidates.append((name, is_2d, plane_size))

    handle.visititems(visitor)
    if not candidates:
        raise RuntimeError("No numeric image-like dataset (ndim >= 2) found in HDF5 file.")

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[0][0]


def load_h5_image(path: Path, dataset_hint: str | None, downsample: int) -> tuple[np.ndarray, str]:
    ds = max(1, int(downsample))
    with h5py.File(path, "r") as handle:
        dataset_name = dataset_hint if (dataset_hint and _dataset_is_image_like(handle, dataset_hint)) else None
        if dataset_name is None:
            dataset_name = choose_image_dataset(handle)

        ds0 = handle[dataset_name]
        if ds0.ndim == 2:
            img = ds0[::ds, ::ds]
        else:
            # Use first frame along all leading dimensions and keep full 2D detector plane.
            index = [0] * (ds0.ndim - 2)
            index.extend([slice(None, None, ds), slice(None, None, ds)])
            img = ds0[tuple(index)]

    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2:
        raise RuntimeError(f"Dataset '{dataset_name}' did not resolve to a 2D image (shape={img.shape}).")
    return img, dataset_name


def image_for_display(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = np.isfinite(image)
    vals = image[finite]
    if vals.size == 0:
        disp = np.zeros_like(image, dtype=np.float32)
        return disp, 0.0, 1.0

    floor = max(float(np.percentile(vals, 1)), 1e-6)
    disp = np.log10(np.clip(image, floor, None))
    vmin = float(np.nanpercentile(disp, 75))
    vmax = float(np.nanpercentile(disp, 99.7))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = float(np.nanmin(disp)), float(np.nanmax(disp))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = 0.0, 1.0
    return disp, vmin, vmax


def launch_simple_viewer(
    root_dir: Path,
    map_npy: Path,
    show_downsample: int = 1,
    xrd_origin: str = "upper",
) -> None:
    map_raw = load_map_npy(map_npy)
    map_disp, map_label = map_for_display(map_raw)

    map_show = map_disp.copy()
    map_ny, map_nx = int(map_show.shape[0]), int(map_show.shape[1])
    index_to_file, ordered_files = sorted_h5_files(root_dir)

    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    ax_map = fig.add_subplot(1, 2, 1)
    ax_img = fig.add_subplot(1, 2, 2)

    im_map = ax_map.imshow(map_show, origin="lower", aspect="auto")
    cbar = fig.colorbar(im_map, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label(map_label)
    ax_map.set_title(f"Click map: {map_npy.name}")
    ax_map.set_xlabel("col")
    ax_map.set_ylabel("row")

    marker = {"artist": None}
    dataset_state = {"dataset_name": None}
    viewer_state = {"xrd_origin": xrd_origin if xrd_origin in {"lower", "upper"} else "lower"}

    def redraw_image(
        row: int,
        col: int,
        row_plot: int,
        col_plot: int,
        expected_index: int,
    ) -> None:
        h5_path = index_to_file.get(expected_index)
        if h5_path is None:
            fallback = expected_index - 1
            if 0 <= fallback < len(ordered_files):
                h5_path = ordered_files[fallback]

        if h5_path is None:
            ax_img.clear()
            ax_img.set_title(
                f"No HDF5 file for map point row={row}, col={col} (expected index={expected_index})"
            )
            ax_img.set_axis_off()
            fig.canvas.draw_idle()
            return

        try:
            img, used_dataset = load_h5_image(
                h5_path,
                dataset_hint=dataset_state["dataset_name"],
                downsample=show_downsample,
            )
            dataset_state["dataset_name"] = used_dataset
        except Exception as e:
            ax_img.clear()
            ax_img.set_title(f"Failed to load image:\n{e}")
            ax_img.set_axis_off()
            fig.canvas.draw_idle()
            return

        disp, vmin, vmax = image_for_display(img)

        ax_img.clear()
        ax_img.imshow(disp, origin=viewer_state["xrd_origin"], aspect="equal", vmin=vmin, vmax=vmax)
        ax_img.set_title(
            f"{h5_path.name} | row={row}, col={col} | idx={expected_index} | "
            f"ds={dataset_state['dataset_name']} | origin={viewer_state['xrd_origin']}"
        )
        ax_img.set_axis_off()

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

        row = row_plot
        col = col_plot
        expected_index = row * map_nx + col + 1

        redraw_image(row, col, row_plot, col_plot, expected_index)

    def onkey(event) -> None:
        if event.key not in ("o", "O"):
            return
        viewer_state["xrd_origin"] = "upper" if viewer_state["xrd_origin"] == "lower" else "lower"
        ax_img.set_title(f"XRD origin changed to {viewer_state['xrd_origin']}. Click a map point to refresh.")
        ax_img.set_axis_off()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)

    plt.tight_layout()
    plt.show(block=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple interactive map -> XRD image viewer.")
    parser.add_argument("--root-dir", type=Path, required=True, help="Root folder containing *_restored.hdf5")
    parser.add_argument("--map-npy", type=Path, required=True, help="2D map NPY file")
    parser.add_argument("--show-downsample", type=int, default=1)
    parser.add_argument("--xrd-origin", choices=["lower", "upper"], default="upper")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launch_simple_viewer(
        root_dir=args.root_dir,
        map_npy=args.map_npy,
        show_downsample=max(1, int(args.show_downsample)),
        xrd_origin=args.xrd_origin,
    )


if __name__ == "__main__":
    main()
