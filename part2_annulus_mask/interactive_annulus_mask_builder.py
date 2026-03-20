#!/usr/bin/env python
"""Interactive annulus-segment mask builder for Dioptas.

Workflow:
1) Load detector center from a pyFAI .poni file (Fit2D centerX/centerY).
2) Load an HDF5 image for visual guidance.
3) Interactively add multiple annulus segments (each with its own phi/chi span).
4) Export a Dioptas-compatible .mask (TIFF with values 0/1).

Modes:
- calibrated: radial bounds are 2theta and azimuth bounds are chi from pyFAI geometry.
- pixel: radial bounds are center-based pixel radii and azimuth bounds are center-based phi.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image


@dataclass
class AnnulusSegment:
    r_in: float
    r_out: float
    phi_start: float
    phi_end: float
    full_circle: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively define concentric annulus masks with phi spans."
    )
    parser.add_argument(
        "--mode",
        choices=["calibrated", "pixel"],
        default="calibrated",
        help=(
            "Mask mode: 'calibrated' uses .poni geometry (2theta/chi), "
            "'pixel' uses center-based circular geometry."
        ),
    )
    parser.add_argument("--poni", required=True, help="Path to pyFAI .poni calibration file.")
    parser.add_argument("--h5", required=True, help="Path to HDF5 image file.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="HDF5 dataset path. If omitted, script auto-selects a likely 2D image dataset.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index for stack datasets (default: 0).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output Dioptas mask path (usually .mask).",
    )
    parser.add_argument(
        "--spec-out",
        default=None,
        help="Optional JSON output for annulus specs (default: out with .json suffix).",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Display lower intensity bound.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Display upper intensity bound.",
    )
    parser.add_argument(
        "--flip-center-y",
        action="store_true",
        help="Legacy option (ignored). Internal processing always uses pre-flipped Y coordinates.",
    )
    parser.add_argument(
        "--display-transform",
        action="append",
        choices=["flipud", "fliplr", "transpose"],
        default=[],
        help=(
            "Legacy option (ignored). Kept only for backward CLI compatibility."
        ),
    )
    return parser.parse_args()


def load_calibration(poni_path: str) -> tuple[Any, tuple[float, float]]:
    import pyFAI

    ai = pyFAI.load(poni_path)
    fit2d = ai.getFit2D()
    center_xy = float(fit2d["centerX"]), float(fit2d["centerY"])
    return ai, center_xy


def build_calibrated_maps(ai: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    tth_deg = np.asarray(ai.center_array(shape=shape, unit="2th_deg"), dtype=np.float32)
    chi_deg = np.asarray(ai.center_array(shape=shape, unit="chi_deg"), dtype=np.float32)
    chi_deg = np.mod(chi_deg, 360.0).astype(np.float32)
    return tth_deg, chi_deg


def _dataset_candidates(handle: h5py.File) -> list[tuple[str, tuple[int, ...], str]]:
    candidates: list[tuple[str, tuple[int, ...], str]] = []

    def visitor(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if obj.ndim < 2:
            return
        if not np.issubdtype(obj.dtype, np.number):
            return
        candidates.append((name, tuple(obj.shape), str(obj.dtype)))

    handle.visititems(visitor)
    return candidates


def choose_dataset(
    handle: h5py.File, requested: str | None
) -> tuple[str, tuple[int, ...], str]:
    candidates = _dataset_candidates(handle)
    if not candidates:
        raise RuntimeError("No numeric dataset with ndim >= 2 found in HDF5 file.")

    if requested:
        if requested not in handle:
            raise KeyError(f"Dataset '{requested}' not found in file.")
        ds = handle[requested]
        if not isinstance(ds, h5py.Dataset) or ds.ndim < 2:
            raise ValueError(f"Dataset '{requested}' is not a numeric image-like dataset.")
        return requested, tuple(ds.shape), str(ds.dtype)

    # Prefer ndim==2, then largest image plane from last two dimensions.
    def score(item: tuple[str, tuple[int, ...], str]) -> tuple[int, int]:
        _, shape, _ = item
        return (1 if len(shape) == 2 else 0, int(shape[-2] * shape[-1]))

    return max(candidates, key=score)


def load_hdf5_image(
    h5_path: str, dataset: str | None, frame: int
) -> tuple[np.ndarray, str, tuple[int, ...], str]:
    with h5py.File(h5_path, "r") as handle:
        ds_path, shape, dtype = choose_dataset(handle, dataset)
        ds = handle[ds_path]

        if ds.ndim == 2:
            image = ds[()]
        else:
            if frame < 0 or frame >= ds.shape[0]:
                raise IndexError(
                    f"Frame {frame} out of range for first dimension size {ds.shape[0]}."
                )
            # For ndim>3, first index uses frame and all remaining leading dims use 0.
            indices: list[Any] = [frame]
            if ds.ndim > 3:
                indices.extend([0] * (ds.ndim - 3))
            indices.extend([slice(None), slice(None)])
            image = ds[tuple(indices)]

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise RuntimeError(
            f"Selected dataset did not resolve to 2D image (got shape {image.shape})."
        )
    return image, ds_path, shape, dtype


def norm_phi(phi_deg: float) -> float:
    return float(phi_deg % 360.0)


def flip_center_y(center_xy: tuple[float, float], height: int) -> tuple[float, float]:
    cx, cy = center_xy
    return float(cx), float((height - 1) - cy)


def angle_between(phi: np.ndarray, start: float, end: float) -> np.ndarray:
    start = norm_phi(start)
    end = norm_phi(end)
    if np.isclose(start, end):
        # Clicking the same angle twice is interpreted as full circle.
        return np.ones_like(phi, dtype=bool)
    if start <= end:
        return (phi >= start) & (phi <= end)
    return (phi >= start) | (phi <= end)


class InteractiveAnnulusMaskBuilder:
    def __init__(
        self,
        image: np.ndarray,
        center_xy: tuple[float, float],
        out_path: Path,
        spec_path: Path,
        metadata: dict[str, Any],
        vmin: float | None,
        vmax: float | None,
        center_y_flipped: bool = False,
        mode: str = "calibrated",
        two_theta_deg_map: np.ndarray | None = None,
        chi_deg_map: np.ndarray | None = None,
        output_unflip_y: bool = False,
        invert_drawn_on_export: bool = True,
    ) -> None:
        self.image = image
        self.height, self.width = image.shape
        self.cx, self.cy = center_xy
        self.center_y_flipped = center_y_flipped
        self.mode = mode
        self.output_unflip_y = output_unflip_y
        self.invert_drawn_on_export = invert_drawn_on_export
        self.out_path = out_path
        self.spec_path = spec_path
        self.metadata = metadata
        self.vmin, self.vmax = self._resolve_display_limits(vmin, vmax)

        if self.mode not in {"pixel", "calibrated"}:
            raise ValueError(f"Unknown mode '{self.mode}'.")
        if self.mode == "calibrated":
            if two_theta_deg_map is None or chi_deg_map is None:
                raise ValueError("Calibrated mode requires two_theta_deg_map and chi_deg_map.")
            if two_theta_deg_map.shape != self.image.shape or chi_deg_map.shape != self.image.shape:
                raise ValueError("Calibration maps must have same shape as image.")

        self.two_theta_deg_map = (
            np.asarray(two_theta_deg_map, dtype=np.float32)
            if two_theta_deg_map is not None
            else None
        )
        self.chi_deg_map = (
            np.asarray(chi_deg_map, dtype=np.float32) if chi_deg_map is not None else None
        )

        self.annuli: list[AnnulusSegment] = []
        self.mask = np.zeros((self.height, self.width), dtype=bool)
        self._dist2: np.ndarray | None = None
        self._phi: np.ndarray | None = None

        self.phase = "idle"
        self.current: dict[str, float] = {}
        self.guide_artists: list[Any] = []

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("Interactive Annulus Mask Builder")

        extent = self._image_extent()
        self.image_artist = self.ax.imshow(
            self.image,
            cmap="gray",
            origin="lower",
            interpolation="nearest",
            extent=extent,
            vmin=self.vmin,
            vmax=self.vmax,
        )
        self.overlay_artist = self.ax.imshow(
            np.zeros((self.height, self.width, 4), dtype=np.float32),
            origin="lower",
            interpolation="nearest",
            extent=extent,
        )

        # Draw both exact sub-pixel center and nearest-pixel center for clarity.
        (self.center_artist,) = self.ax.plot(
            [self.cx],
            [self.cy],
            marker="+",
            markersize=12,
            color="cyan",
            mew=2,
            linestyle="None",
        )
        (self.center_dot_artist,) = self.ax.plot(
            [self.cx],
            [self.cy],
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor="cyan",
            mew=1.4,
            linestyle="None",
        )
        (self.center_pixel_artist,) = self.ax.plot(
            [round(self.cx)],
            [round(self.cy)],
            marker="x",
            markersize=7,
            color="yellow",
            mew=1.4,
            linestyle="None",
        )
        (self.center_hline_artist,) = self.ax.plot(
            [self.cx - 18, self.cx + 18],
            [self.cy, self.cy],
            color="cyan",
            lw=1.1,
        )
        (self.center_vline_artist,) = self.ax.plot(
            [self.cx, self.cx],
            [self.cy - 18, self.cy + 18],
            color="cyan",
            lw=1.1,
        )
        self.tth_min_xy: tuple[float, float] | None = None
        self.tth_min_artist = None
        if self.mode == "calibrated":
            self._update_tth_min_reference()
        self.ax.set_title("HDF5 image with annulus-segment mask overlay")
        self.ax.set_aspect("equal", adjustable="box")
        self.display_transforms: list[str] = []
        self._update_axes_limits()

        self.status_text = self.fig.text(
            0.01,
            0.01,
            "",
            ha="left",
            va="bottom",
            family="monospace",
            fontsize=10,
        )

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self._set_status(self._controls_text())

    def _resolve_display_limits(
        self, vmin: float | None, vmax: float | None
    ) -> tuple[float, float]:
        finite = self.image[np.isfinite(self.image)]
        if finite.size == 0:
            return 0.0, 1.0
        if vmin is None:
            vmin = float(np.percentile(finite, 1.0))
        if vmax is None:
            vmax = float(np.percentile(finite, 99.5))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        return float(vmin), float(vmax)

    def _controls_text(self) -> str:
        return (
            "Controls: n=new annulus | f=full 360 (after radii) | u=undo | c=clear | "
            "s=save mask | j=save json | q=quit"
        )

    def _radial_unit_label(self) -> str:
        return "deg(2theta)" if self.mode == "calibrated" else "px"

    def _azimuth_label(self) -> str:
        return "chi" if self.mode == "calibrated" else "phi"

    def _annuli_summary(self) -> str:
        if not self.annuli:
            return "Annuli: none"
        lines = [f"Annuli: {len(self.annuli)}"]
        for idx, a in enumerate(self.annuli, start=1):
            if a.full_circle:
                phi_txt = "0->360"
            else:
                phi_txt = f"{a.phi_start:.1f}->{a.phi_end:.1f}"
            lines.append(
                f"{idx:02d}: r={a.r_in:.3f}-{a.r_out:.3f} {self._radial_unit_label()}  "
                f"{self._azimuth_label()}={phi_txt} deg"
            )
        return "\n".join(lines[-10:])

    def _set_status(self, extra: str = "") -> None:
        phase_map = {
            "idle": "idle (press n to add annulus)",
            "r_in": "click INNER radius",
            "r_out": "click OUTER radius",
            "phi_start": "click PHI START angle",
            "phi_end": "click PHI END angle",
        }
        phase = phase_map.get(self.phase, self.phase)
        transforms_txt = " -> ".join(self.display_transforms) if self.display_transforms else "none"
        center_ix = int(round(self.cx))
        center_iy = int(round(self.cy))
        tth_txt = ""
        if self.tth_min_xy is not None:
            tmx, tmy = self.tth_min_xy
            tth_txt = f"2theta-min pixel (x,y)=({tmx:.1f}, {tmy:.1f}) [magenta x]\n"
        text = (
            f"Mode: {self.mode}\n"
            f"Center (x,y)=({self.cx:.3f}, {self.cy:.3f})\n"
            f"Nearest center pixel (x,y)=({center_ix}, {center_iy}) [yellow x]\n"
            f"{tth_txt}"
            f"Internal coordinates use mandatory Y flip from raw image: {self.center_y_flipped}\n"
            f"Drawn region: KEEP | Exported Dioptas mask: IGNORE (inverse)\n"
            f"Display transforms: {transforms_txt}\n"
            f"Phase: {phase}\n"
            f"{self._annuli_summary()}\n"
            f"{self._controls_text()}"
        )
        if extra:
            text += f"\n{extra}"
        self.status_text.set_text(text)
        self.fig.canvas.draw_idle()

    def _angle_deg(self, x: float, y: float) -> float:
        if self.mode == "calibrated":
            assert self.chi_deg_map is not None
            iy = int(np.clip(round(y), 0, self.height - 1))
            ix = int(np.clip(round(x), 0, self.width - 1))
            return norm_phi(float(self.chi_deg_map[iy, ix]))
        # In image pixel coordinates (y increases downward).
        return norm_phi(np.degrees(np.arctan2(y - self.cy, x - self.cx)))

    def _radius_px(self, x: float, y: float) -> float:
        if self.mode == "calibrated":
            assert self.two_theta_deg_map is not None
            iy = int(np.clip(round(y), 0, self.height - 1))
            ix = int(np.clip(round(x), 0, self.width - 1))
            return float(self.two_theta_deg_map[iy, ix])
        return float(np.hypot(x - self.cx, y - self.cy))

    def _ensure_geometry_cache(self) -> None:
        if self.mode != "pixel":
            return
        if self._dist2 is not None and self._phi is not None:
            return
        yy, xx = np.indices((self.height, self.width), dtype=np.float32)
        self._dist2 = (xx - self.cx) ** 2 + (yy - self.cy) ** 2
        self._phi = (np.degrees(np.arctan2(yy - self.cy, xx - self.cx)) + 360.0) % 360.0

    def _set_center(self, cx: float, cy: float) -> None:
        self.cx = float(cx)
        self.cy = float(cy)
        self._dist2 = None
        self._phi = None
        self.center_artist.set_data([self.cx], [self.cy])
        self.center_dot_artist.set_data([self.cx], [self.cy])
        self.center_pixel_artist.set_data([round(self.cx)], [round(self.cy)])
        self.center_hline_artist.set_data([self.cx - 18, self.cx + 18], [self.cy, self.cy])
        self.center_vline_artist.set_data([self.cx, self.cx], [self.cy - 18, self.cy + 18])
        self._update_tth_min_reference()
        self._draw_guides()
        self._refresh_overlay()

    def _update_axes_limits(self) -> None:
        # Pixel centers are at integer coordinates with origin='lower'.
        self.ax.set_xlim(-0.5, self.width - 0.5)
        self.ax.set_ylim(-0.5, self.height - 0.5)

    def _image_extent(self) -> tuple[float, float, float, float]:
        return (-0.5, self.width - 0.5, -0.5, self.height - 0.5)

    def _update_tth_min_reference(self) -> None:
        if self.mode != "calibrated" or self.two_theta_deg_map is None:
            return
        iy, ix = np.unravel_index(np.nanargmin(self.two_theta_deg_map), self.two_theta_deg_map.shape)
        self.tth_min_xy = (float(ix), float(iy))
        if self.tth_min_artist is None:
            (self.tth_min_artist,) = self.ax.plot(
                [ix],
                [iy],
                marker="x",
                markersize=7,
                color="magenta",
                mew=1.5,
                linestyle="None",
            )
        else:
            self.tth_min_artist.set_data([ix], [iy])

    def _apply_display_transform(self, op: str) -> None:
        old_h, old_w = self.height, self.width
        new_cx, new_cy = self.cx, self.cy
        if op == "flipud":
            self.image = np.flipud(self.image)
            if self.two_theta_deg_map is not None:
                self.two_theta_deg_map = np.flipud(self.two_theta_deg_map)
            if self.chi_deg_map is not None:
                self.chi_deg_map = np.flipud(self.chi_deg_map)
            new_cy = (old_h - 1) - self.cy
        elif op == "fliplr":
            self.image = np.fliplr(self.image)
            if self.two_theta_deg_map is not None:
                self.two_theta_deg_map = np.fliplr(self.two_theta_deg_map)
            if self.chi_deg_map is not None:
                self.chi_deg_map = np.fliplr(self.chi_deg_map)
            new_cx = (old_w - 1) - self.cx
        elif op == "transpose":
            self.image = self.image.T
            if self.two_theta_deg_map is not None:
                self.two_theta_deg_map = self.two_theta_deg_map.T
            if self.chi_deg_map is not None:
                self.chi_deg_map = self.chi_deg_map.T
            new_cx, new_cy = self.cy, self.cx
        else:
            raise ValueError(f"Unknown display transform op: {op}")

        self.height, self.width = self.image.shape
        self.cx, self.cy = float(new_cx), float(new_cy)
        self._dist2 = None
        self._phi = None
        self.center_artist.set_data([self.cx], [self.cy])
        self.center_dot_artist.set_data([self.cx], [self.cy])
        self.center_pixel_artist.set_data([round(self.cx)], [round(self.cy)])
        self.center_hline_artist.set_data([self.cx - 18, self.cx + 18], [self.cy, self.cy])
        self.center_vline_artist.set_data([self.cx, self.cx], [self.cy - 18, self.cy + 18])
        self._update_tth_min_reference()
        self.image_artist.set_data(self.image)
        self.image_artist.set_extent(self._image_extent())
        self.overlay_artist.set_extent(self._image_extent())
        self._update_axes_limits()
        self.display_transforms.append(op)
        self._draw_guides()
        self._refresh_overlay()
        self._set_status(f"Applied display transform: {op}")

    def toggle_center_y_flip(self) -> None:
        self.center_y_flipped = not self.center_y_flipped
        if self.mode == "calibrated":
            assert self.two_theta_deg_map is not None
            assert self.chi_deg_map is not None
            self.two_theta_deg_map = np.flipud(self.two_theta_deg_map)
            self.chi_deg_map = np.flipud(self.chi_deg_map)
            self._draw_guides()
            self._refresh_overlay()
            self._set_status("Toggled calibration-map Y flip.")
            return
        self._set_center(self.cx, (self.height - 1) - self.cy)
        self._set_status("Toggled center Y flip.")

    def _build_mask(self) -> np.ndarray:
        self._ensure_geometry_cache()
        mask = np.zeros((self.height, self.width), dtype=bool)
        if self.mode == "pixel":
            assert self._dist2 is not None
            assert self._phi is not None
            radial_map = self._dist2
            azimuth_map = self._phi
        else:
            assert self.two_theta_deg_map is not None
            assert self.chi_deg_map is not None
            radial_map = self.two_theta_deg_map
            azimuth_map = self.chi_deg_map

        for a in self.annuli:
            if self.mode == "pixel":
                radial = (radial_map >= a.r_in**2) & (radial_map <= a.r_out**2)
            else:
                radial = np.isfinite(radial_map) & (radial_map >= a.r_in) & (radial_map <= a.r_out)
            angular = (
                np.ones_like(mask, dtype=bool)
                if a.full_circle
                else angle_between(azimuth_map, a.phi_start, a.phi_end)
            )
            mask |= radial & angular
        return mask

    def _refresh_overlay(self) -> None:
        self.mask = self._build_mask()
        rgba = np.zeros((self.height, self.width, 4), dtype=np.float32)
        # Green overlay = region you are selecting to KEEP.
        rgba[..., 1] = 1.0
        rgba[..., 3] = self.mask.astype(np.float32) * 0.35
        self.overlay_artist.set_data(rgba)
        self.fig.canvas.draw_idle()

    def _clear_guides(self) -> None:
        for artist in self.guide_artists:
            if hasattr(artist, "remove"):
                artist.remove()
            elif hasattr(artist, "collections"):
                for coll in artist.collections:
                    coll.remove()
        self.guide_artists.clear()

    def _add_contour_guide(self, arr: np.ndarray, level: float, color: str) -> None:
        if not np.isfinite(level):
            return
        try:
            # Use explicit pixel-center coordinates so contour and marker share
            # the exact same coordinate system.
            x = np.arange(self.width, dtype=np.float32)
            y = np.arange(self.height, dtype=np.float32)
            contour = self.ax.contour(
                x,
                y,
                arr,
                levels=[float(level)],
                colors=[color],
                linewidths=1.2,
                linestyles="--",
            )
            self.guide_artists.append(contour)
        except Exception:
            # Keep UI responsive even if contouring fails on pathological level values.
            pass

    def _draw_guides(self) -> None:
        self._clear_guides()
        r_in = self.current.get("r_in")
        r_out = self.current.get("r_out")
        phi_start = self.current.get("phi_start")
        phi_end = self.current.get("phi_end")
        max_r = float(np.hypot(self.width, self.height))

        if self.mode == "pixel":
            if r_in is not None:
                c = Circle((self.cx, self.cy), r_in, fill=False, ls="--", lw=1.2, color="cyan")
                self.ax.add_patch(c)
                self.guide_artists.append(c)
            if r_out is not None:
                c = Circle((self.cx, self.cy), r_out, fill=False, ls="--", lw=1.2, color="yellow")
                self.ax.add_patch(c)
                self.guide_artists.append(c)

            for angle, color in ((phi_start, "lime"), (phi_end, "orange")):
                if angle is None:
                    continue
                theta = np.deg2rad(angle)
                x2 = self.cx + max_r * np.cos(theta)
                y2 = self.cy + max_r * np.sin(theta)
                (line,) = self.ax.plot(
                    [self.cx, x2], [self.cy, y2], ls="--", lw=1.2, color=color
                )
                self.guide_artists.append(line)
            return

        # Calibrated mode guides: contours of constant 2theta / chi.
        assert self.two_theta_deg_map is not None
        assert self.chi_deg_map is not None
        if r_in is not None:
            self._add_contour_guide(self.two_theta_deg_map, r_in, "cyan")
        if r_out is not None:
            self._add_contour_guide(self.two_theta_deg_map, r_out, "yellow")

        for angle, color in ((phi_start, "lime"), (phi_end, "orange")):
            if angle is None:
                continue
            self._add_contour_guide(self.chi_deg_map, angle, color)

    def start_new_annulus(self) -> None:
        self.phase = "r_in"
        self.current.clear()
        self._draw_guides()
        self._set_status(
            "New annulus: click inner radial bound, outer radial bound, "
            f"{self._azimuth_label()} start, {self._azimuth_label()} end."
        )

    def on_click(self, event: Any) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)
        r = self._radius_px(x, y)
        phi = self._angle_deg(x, y)
        if self.phase == "idle":
            msg = (
                f"Cursor x,y=({x:.3f},{y:.3f})  "
                f"{self._radial_unit_label()}={r:.3f}  {self._azimuth_label()}={phi:.2f} deg"
            )
            self._set_status(msg)
            print(msg)
            return

        if self.phase == "r_in":
            self.current["r_in"] = r
            self.phase = "r_out"
            self._draw_guides()
            self._set_status(
                f"Inner radial bound = {r:.3f} {self._radial_unit_label()}. "
                "Click outer bound."
            )
            return

        if self.phase == "r_out":
            r_in = self.current["r_in"]
            self.current["r_in"] = min(r_in, r)
            self.current["r_out"] = max(r_in, r)
            self.phase = "phi_start"
            self._draw_guides()
            self._set_status(
                f"Outer radial bound = {self.current['r_out']:.3f} {self._radial_unit_label()}. "
                f"Click {self._azimuth_label()} start."
            )
            return

        if self.phase == "phi_start":
            self.current["phi_start"] = phi
            self.phase = "phi_end"
            self._draw_guides()
            self._set_status(
                f"{self._azimuth_label()} start = {phi:.2f} deg. Click {self._azimuth_label()} end."
            )
            return

        if self.phase == "phi_end":
            self.current["phi_end"] = phi
            self._finalize_current(full_circle=False)

    def _finalize_current(self, full_circle: bool) -> None:
        r_in = float(self.current["r_in"])
        r_out = float(self.current["r_out"])
        if full_circle:
            segment = AnnulusSegment(r_in=r_in, r_out=r_out, phi_start=0.0, phi_end=360.0, full_circle=True)
        else:
            segment = AnnulusSegment(
                r_in=r_in,
                r_out=r_out,
                phi_start=float(self.current["phi_start"]),
                phi_end=float(self.current["phi_end"]),
                full_circle=False,
            )
        self.annuli.append(segment)
        self.phase = "idle"
        self.current.clear()
        self._draw_guides()
        self._refresh_overlay()
        self._set_status(
            "Added annulus "
            f"#{len(self.annuli)}: r={segment.r_in:.3f}-{segment.r_out:.3f} {self._radial_unit_label()}"
        )

    def undo(self) -> None:
        if self.phase != "idle":
            self.phase = "idle"
            self.current.clear()
            self._draw_guides()
            self._set_status("Canceled in-progress annulus.")
            return
        if not self.annuli:
            self._set_status("Nothing to undo.")
            return
        self.annuli.pop()
        self._refresh_overlay()
        self._set_status("Removed last annulus.")

    def clear_all(self) -> None:
        self.phase = "idle"
        self.current.clear()
        self.annuli.clear()
        self._draw_guides()
        self._refresh_overlay()
        self._set_status("Cleared all annuli.")

    def save_mask(self) -> None:
        out = self.out_path
        out.parent.mkdir(parents=True, exist_ok=True)
        mask_out = self.mask
        if self.invert_drawn_on_export:
            mask_out = ~mask_out
        export_y_flips = 2 if self.output_unflip_y else 0
        for _ in range(export_y_flips):
            mask_out = np.flipud(mask_out)
        mask_i32 = mask_out.astype(np.int32)
        Image.fromarray(mask_i32).save(out, format="TIFF", compression="tiff_deflate")
        self._set_status(
            f"Saved mask: {out} (masked pixels={int(mask_i32.sum())}, "
            f"invert_drawn_on_export={self.invert_drawn_on_export}, export_y_flips={export_y_flips})"
        )
        print(f"Saved mask: {out}")
        print(f"Masked pixels (value=1): {int(mask_i32.sum())}")

    def save_spec(self) -> None:
        payload = {
            "mode": self.mode,
            "radial_unit": "2theta_deg" if self.mode == "calibrated" else "pixel_radius",
            "azimuth_unit": "chi_deg" if self.mode == "calibrated" else "phi_deg",
            "center_xy_fit2d_px": [self.cx, self.cy],
            "center_xy_nearest_pixel": [int(round(self.cx)), int(round(self.cy))],
            "center_y_flipped": self.center_y_flipped,
            "output_unflip_y": self.output_unflip_y,
            "invert_drawn_on_export": self.invert_drawn_on_export,
            "export_y_flips": 2 if self.output_unflip_y else 0,
            "display_transforms": list(self.display_transforms),
            "shape_yx": [self.height, self.width],
            "annuli": [asdict(a) for a in self.annuli],
            "metadata": self.metadata,
        }
        self.spec_path.parent.mkdir(parents=True, exist_ok=True)
        with self.spec_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        self._set_status(f"Saved spec JSON: {self.spec_path}")
        print(f"Saved spec JSON: {self.spec_path}")

    def on_key_press(self, event: Any) -> None:
        key = (event.key or "").lower()
        if key == "n":
            self.start_new_annulus()
        elif key == "f":
            if self.phase in {"phi_start", "phi_end"} and "r_in" in self.current and "r_out" in self.current:
                self._finalize_current(full_circle=True)
            else:
                self._set_status("Press f after choosing inner/outer radius for a full 360 annulus.")
        elif key in {"u", "backspace"}:
            self.undo()
        elif key == "c":
            self.clear_all()
        elif key == "s":
            self.save_mask()
        elif key == "j":
            self.save_spec()
        elif key in {"q", "escape"}:
            plt.close(self.fig)
        elif key == "h":
            self._set_status(self._controls_text())

    def run(self) -> None:
        self._refresh_overlay()
        self._set_status(self._controls_text())
        plt.tight_layout()
        plt.show()


def main() -> None:
    args = parse_args()

    out_path = Path(args.out)
    spec_path = Path(args.spec_out) if args.spec_out else out_path.with_suffix(".json")

    ai, center_xy_from_poni = load_calibration(args.poni)
    image_raw, ds_path, ds_shape, ds_dtype = load_hdf5_image(args.h5, args.dataset, args.frame)
    # Canonical internal coordinates: pre-flip Y once, then all geometry/math uses this frame.
    image = np.flipud(image_raw)
    two_theta_deg_map: np.ndarray | None = None
    chi_deg_map: np.ndarray | None = None
    y_flip_applied = True
    if args.mode == "calibrated":
        # Bring calibration maps into the same internal y-flipped coordinate frame.
        two_theta_deg_map, chi_deg_map = build_calibrated_maps(ai, image.shape)
        two_theta_deg_map = np.flipud(two_theta_deg_map)
        chi_deg_map = np.flipud(chi_deg_map)

    # Center values from calibration are already in the chosen internal plotting orientation.
    center_xy = center_xy_from_poni

    metadata = {
        "mode": args.mode,
        "poni": str(Path(args.poni).resolve()),
        "h5": str(Path(args.h5).resolve()),
        "dataset_path": ds_path,
        "dataset_shape": list(ds_shape),
        "dataset_dtype": ds_dtype,
        "frame": int(args.frame),
        "center_from_poni_fit2d_xy": [center_xy_from_poni[0], center_xy_from_poni[1]],
        "center_internal_xy_used_directly": [center_xy[0], center_xy[1]],
        "y_flip_applied_initially": y_flip_applied,
        "center_transformed": False,
        "phi_convention_deg": (
            "chi_deg from pyFAI geometry (wrapped to [0,360))"
            if args.mode == "calibrated"
            else "atan2(y-cy, x-cx) in image pixel coordinates; wraps [0,360)"
        ),
        "radial_convention": (
            "2theta_deg from pyFAI center_array"
            if args.mode == "calibrated"
            else "pixel distance from center"
        ),
        "mask_convention": "1=masked, 0=unmasked",
    }

    print("Loaded image:")
    print(f"  h5: {args.h5}")
    print(f"  dataset: {ds_path} shape={ds_shape} dtype={ds_dtype}")
    print(f"  raw image shape: {image_raw.shape}")
    print(f"  internal (y-flipped) image shape: {image.shape}")
    print(f"Mode: {args.mode}")
    print(
        "Center from .poni (Fit2D): "
        f"x={center_xy_from_poni[0]:.3f}, y={center_xy_from_poni[1]:.3f}"
    )
    print(
        "Center used in internal/view coordinates (no center transform): "
        f"x={center_xy[0]:.3f}, y={center_xy[1]:.3f}"
    )
    if args.flip_center_y:
        print("Note: --flip-center-y is ignored (internal y-flip is always applied).")
    if args.display_transform:
        print("Note: --display-transform is ignored to keep coordinate system consistent.")
    print("Interactive controls:")
    if args.mode == "calibrated":
        print("  n=new annulus, click 2theta_in -> 2theta_out -> chi_start -> chi_end")
    else:
        print("  n=new annulus, click r_in -> r_out -> phi_start -> phi_end")
    print("  f=full 360 annulus (after radii)")
    print("  u=undo, c=clear, s=save mask, j=save json, q=quit")
    print("  Mask export applies an extra Y-flip correction for Dioptas import orientation.")

    app = InteractiveAnnulusMaskBuilder(
        image=image,
        center_xy=center_xy,
        out_path=out_path,
        spec_path=spec_path,
        metadata=metadata,
        vmin=args.vmin,
        vmax=args.vmax,
        center_y_flipped=y_flip_applied,
        mode=args.mode,
        two_theta_deg_map=two_theta_deg_map,
        chi_deg_map=chi_deg_map,
        output_unflip_y=True,
        invert_drawn_on_export=True,
    )
    app.run()


if __name__ == "__main__":
    main()
