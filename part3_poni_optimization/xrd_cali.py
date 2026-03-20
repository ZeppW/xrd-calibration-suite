#%%
import numpy as np
import h5py
import pyFAI
import matplotlib.pyplot as plt
import copy
from pathlib import Path
from PIL import Image

def load_image_h5(path_h5, dataset="data", frame=0, downsample=1):
    with h5py.File(path_h5, "r") as f:
        ds = f[dataset]
        if ds.ndim == 3 and ds.shape[0] == 1:
            img = ds[0]
        elif ds.ndim == 3:
            img = ds[frame]
        else:
            img = ds[...]
    img = np.array(img, dtype=np.float32)
    if downsample > 1:
        img = img[::downsample, ::downsample]
    return img

def load_ai_with_downsample(poni_path, downsample=1):
    ai = pyFAI.load(poni_path)
    if downsample > 1:
        ai = copy.deepcopy(ai)
        ai.detector.pixel1 *= downsample
        ai.detector.pixel2 *= downsample
    return ai


def _load_mask_any(path, expected_shape=None, npz_key=None):
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        npz = np.load(path, allow_pickle=True)
        keys = list(npz.keys())
        if npz_key:
            if npz_key not in npz:
                raise KeyError(f"npz key '{npz_key}' not found in {path}. Keys: {keys}")
            arr = npz[npz_key]
        else:
            # Prefer a key containing "mask".
            key = next((k for k in keys if "mask" in k.lower()), keys[0])
            arr = npz[key]
    else:
        arr = np.array(Image.open(path))
        if arr.ndim == 3:
            # For RGBA mask PNGs, alpha is usually the intended binary support.
            if arr.shape[2] >= 4 and np.unique(arr[..., 3]).size > 1:
                arr = arr[..., 3]
            else:
                arr = arr.max(axis=2)

    arr = np.asarray(arr)
    if expected_shape is not None and arr.shape != tuple(expected_shape):
        # Common accidental transpose case.
        if arr.ndim == 2 and arr.T.shape == tuple(expected_shape):
            arr = arr.T
        else:
            raise ValueError(
                f"Mask shape {arr.shape} does not match expected image shape {tuple(expected_shape)}."
            )
    return arr


def _resolve_mask_source(mask_source, expected_shape=None, npz_key=None):
    """Resolve file or directory mask source to a boolean mask array."""
    if mask_source is None:
        return None, None

    p = Path(mask_source)
    if p.is_file():
        arr = _load_mask_any(p, expected_shape=expected_shape, npz_key=npz_key)
        return arr, p

    if not p.is_dir():
        raise FileNotFoundError(f"Mask source does not exist: {p}")

    exts = {".mask", ".tif", ".tiff", ".png", ".npy", ".npz"}
    candidates = []
    for f in p.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in exts:
            continue
        if "mask" not in f.name.lower():
            continue
        candidates.append(f)

    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No mask-like files found under: {p}")

    # Prefer the newest file that matches expected shape.
    for f in candidates:
        try:
            arr = _load_mask_any(f, expected_shape=expected_shape, npz_key=npz_key)
            return arr, f
        except Exception:
            continue
    # If none matched shape, surface the newest for a clear error.
    arr = _load_mask_any(candidates[0], expected_shape=expected_shape, npz_key=npz_key)
    return arr, candidates[0]


def integrate_cake_2d(
    img,
    poni_path=None,
    ai=None,
    npt_rad=2000,
    npt_azim=360,
    unit="2th_deg",
    radial_range=None,
    azimuth_range=None,
    mask=None,
    mask_source=None,
    mask_npz_key=None,
    mask_is_keep_region=False,
    mask_transform=None,
    correct_solid_angle=True,
    polarization_factor=None,
    azimuth_to_0_360=True,
    phi_range_deg=None,
    method=("bbox", "csr", "cython"),
    return_info=False,
):
    """Compute 2D cake integration for downstream analysis.

    Returns
    -------
    cake : ndarray, shape (npt_azim, npt_rad)
        2D intensity map.
    radial : ndarray, shape (npt_rad,)
        Radial bins in the requested `unit`.
    phi : ndarray, shape (npt_azim,)
        Azimuth/phi bins in degrees.
    """
    if ai is None:
        if poni_path is None:
            raise ValueError("Provide either `ai` or `poni_path`.")
        ai = pyFAI.load(poni_path)

    img = np.asarray(img, dtype=np.float32)
    if mask is not None and mask_source is not None:
        raise ValueError("Use only one of `mask` or `mask_source`.")

    resolved_mask_path = None
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
    elif mask_source is not None:
        mask_raw, resolved_mask_path = _resolve_mask_source(
            mask_source, expected_shape=img.shape, npz_key=mask_npz_key
        )
        mask_arr = np.asarray(mask_raw != 0, dtype=bool)
    else:
        mask_arr = None

    # Optional alignment transforms on the mask before applying semantics.
    if mask_arr is not None and mask_transform:
        ops = [mask_transform] if isinstance(mask_transform, str) else list(mask_transform)
        for op in ops:
            if op == "flipud":
                mask_arr = np.flipud(mask_arr)
            elif op == "fliplr":
                mask_arr = np.fliplr(mask_arr)
            elif op == "transpose":
                mask_arr = mask_arr.T
            else:
                raise ValueError(
                    f"Unknown mask_transform op '{op}'. Use flipud/fliplr/transpose."
                )

    # pyFAI convention: mask=True means IGNORE this pixel.
    # If user provides keep-region mask, invert before integration.
    if mask_arr is not None and mask_is_keep_region:
        mask_arr = ~mask_arr

    az_range = phi_range_deg if phi_range_deg is not None else azimuth_range

    res = ai.integrate2d(
        img,
        npt_rad=npt_rad,
        npt_azim=npt_azim,
        unit=unit,
        radial_range=radial_range,
        azimuth_range=az_range,
        mask=mask_arr,
        correctSolidAngle=bool(correct_solid_angle),
        polarization_factor=polarization_factor,
        method=method,
    )

    cake = np.asarray(res.intensity, dtype=np.float32)
    radial = np.asarray(res.radial, dtype=np.float32)
    phi = np.asarray(res.azimuthal, dtype=np.float32)

    # Optional: remap azimuth from [-180,180) to [0,360) and reorder rows.
    if azimuth_to_0_360:
        phi = np.mod(phi + 360.0, 360.0)
        order = np.argsort(phi)
        phi = phi[order]
        cake = cake[order, :]

    if not return_info:
        return cake, radial, phi

    info = {
        "unit": unit,
        "npt_rad": int(npt_rad),
        "npt_azim": int(npt_azim),
        "phi_range_deg": phi_range_deg if phi_range_deg is not None else azimuth_range,
        "mask_source": str(mask_source) if mask_source is not None else None,
        "resolved_mask_path": str(resolved_mask_path) if resolved_mask_path is not None else None,
        "mask_is_keep_region": bool(mask_is_keep_region),
        "mask_transform": mask_transform,
    }
    return cake, radial, phi, info


def save_cake_npz(path_out, cake, radial, azimuthal, unit="2th_deg", extra_meta=None):
    """Save cake data as compressed NPZ for later analysis."""
    path_out = Path(path_out)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"unit": unit}
    if extra_meta:
        meta.update(dict(extra_meta))

    np.savez_compressed(
        path_out,
        cake=np.asarray(cake, dtype=np.float32),
        radial=np.asarray(radial, dtype=np.float32),
        azimuthal=np.asarray(azimuthal, dtype=np.float32),
        metadata=meta,
    )
    return path_out


def plot_cake_2d(
    cake,
    two_theta,
    phi,
    out_png=None,
    log_scale=True,
    title="Cake 2D (x=2theta, y=phi)",
    cmap="gray",
):
    """Plot cake with explicit axis convention: x=2theta, y=phi."""
    cake = np.asarray(cake, dtype=np.float32)
    two_theta = np.asarray(two_theta, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)

    Z = np.log10(np.maximum(cake, 1e-6)) if log_scale else cake
    extent = [float(two_theta[0]), float(two_theta[-1]), float(phi[0]), float(phi[-1])]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent, cmap=cmap)
    ax.set_xlabel("2theta (deg)")
    ax.set_ylabel("phi (deg)")
    ax.set_title(title + (" (log10)" if log_scale else ""))
    fig.colorbar(im, ax=ax, label="intensity" + (" (log10)" if log_scale else ""))
    fig.tight_layout()

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=180)
    return fig, ax

def theta_chi_map(img, ai, n_tth=1200, n_chi=720,
                  tth_range_deg=None, chi_range_deg=(-180, 180),
                  ignore_value=-1, use_mean=True):
    H, W = img.shape
    bad = (~np.isfinite(img)) | (img == ignore_value)

    # Recommended modern API:
    tth = ai.center_array(shape=(H, W), unit="2th_rad")   # 2θ in radians
    chi = ai.center_array(shape=(H, W), unit="chi_rad")   # χ in radians

    tth_deg = np.degrees(tth)
    chi_deg = np.degrees(chi)

    m = ~bad
    x = tth_deg[m].ravel()
    y = chi_deg[m].ravel()
    w = img[m].ravel().astype(np.float64)

    if tth_range_deg is None:
        lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
        tth_range_deg = (float(lo), float(hi))

    tth_edges = np.linspace(tth_range_deg[0], tth_range_deg[1], n_tth + 1)
    chi_edges = np.linspace(chi_range_deg[0], chi_range_deg[1], n_chi + 1)

    sums, _, _ = np.histogram2d(y, x, bins=[chi_edges, tth_edges], weights=w)
    counts, _, _ = np.histogram2d(y, x, bins=[chi_edges, tth_edges])

    I2d = sums / np.maximum(counts, 1) if use_mean else sums

    tth_centers = 0.5 * (tth_edges[:-1] + tth_edges[1:])
    chi_centers = 0.5 * (chi_edges[:-1] + chi_edges[1:])

    return tth_centers, chi_centers, I2d

def show_theta_chi(img, poni_path, downsample=1,
                   n_tth=1000, n_chi=720,
                   tth_range_deg=None, chi_range_deg=(-180, 180),
                   ignore_value=-1, log_scale=True):
    ai = load_ai_with_downsample(poni_path, downsample=downsample)
    tth_axis, chi_axis, I2d = theta_chi_map(
        img, ai, n_tth=n_tth, n_chi=n_chi,
        tth_range_deg=tth_range_deg, chi_range_deg=chi_range_deg,
        ignore_value=ignore_value, use_mean=True
    )

    Z = np.log10(np.maximum(I2d, 1e-3)) if log_scale else I2d

    plt.figure(figsize=(10, 5))
    extent = [tth_axis[0], tth_axis[-1], chi_axis[0], chi_axis[-1]]
    plt.imshow(Z, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("2θ (deg)")
    plt.ylabel("χ (deg)")
    plt.title("2D viewer: 2θ vs χ" + (" (log10)" if log_scale else ""))
    plt.colorbar(label="mean intensity" + (" (log10)" if log_scale else ""))
    plt.tight_layout()
    plt.show()

    return tth_axis, chi_axis, I2d
if __name__ == "__main__":
    # Minimal example (edit paths before running this file directly).
    poni_path = "E:/XRD/proc/proc/LaB6_003_25keV_poni.poni"
    path_h5 = "E:/XRD/data/PIMEGA_gridscan_00_47_31_restored_files/PIMEGA_gridscan_00_47_31_521_restored.hdf5"
    if Path(path_h5).exists() and Path(poni_path).exists():
        img = load_image_h5(path_h5, dataset="data", frame=0, downsample=1)
        cake, tth, phi = integrate_cake_2d(
            img,
            poni_path=poni_path,
            npt_rad=1200,
            npt_azim=720,
            unit="2th_deg",
            azimuth_to_0_360=True,
        )
        plot_cake_2d(cake, tth, phi, log_scale=True)
        plt.show()
