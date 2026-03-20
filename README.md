# XRD Calibration Suite

This repo provides 3 tools used in your workflow:

1. `part1_map_viewer`: click a map point and view the corresponding XRD image.
2. `part2_annulus_mask`: interactively draw annulus/phi masks and export Dioptas `.mask`.
3. `part3_poni_optimization`: optimize PONI (Fit2D center/tilt space).

## Quick Start

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Part 1: Map -> XRD Image Viewer

### Inputs

- Root dataset folder containing `*_restored.hdf5`
- Custom 2D map `.npy`

### Run (GUI)

```bash
python part1_map_viewer/launch_map_viewer_gui.py
```

### Run (CLI)

```bash
python part1_map_viewer/view_map_with_1d.py \
  --root-dir E:/XRD/data/PIMEGA_gridscan_00_47_31_restored_files \
  --map-npy F:/NMR/NMR/py_projects/xrd/my_custom_map.npy
```

### Notes

- Left panel: map (`origin=lower`).
- Right panel: XRD image (`imshow`), default `origin=upper`.
- Press `o` to toggle right-image origin (`upper`/`lower`).

## Part 2: Annulus Mask Builder

### Recommended launcher (simple)

```bash
python part2_annulus_mask/launch_mask_maker_gui.py
```

You only choose:
- `.poni` file
- `.h5/.hdf5` image file
- save directory for outputs

Outputs:
- `<h5_stem>.mask`
- `<h5_stem>.json`

### Legacy full launcher

```bash
python part2_annulus_mask/launch_annulus_builder_gui.py
```

### Interactive keys

- `n`: start new annulus
- `f`: full 360 annulus (after inner/outer radius)
- `u`: undo
- `c`: clear all
- `s`: save mask directly to selected output path
- `j`: save JSON directly to selected output path
- `q`: quit

Note: Matplotlib default save-dialog shortcut is disabled in this tool, so `s/j` always use project save paths.

## Part 3: PONI Optimization

Main scripts:
- `poni_theta_phi_check.py`: top-point selection, cake generation, theta-vs-phi checks
- `poni_grid_search_4d.py`: coarse grid search in `dx`, `dy`, `dtilt`, `dtiltplan`
- `poni_local_refine.py`: local coordinate-descent refinement

Current project conventions:
- skip `col=0` during PONI calibration point selection
- robust row-fit filtering enabled
- optional essential top-20 validation
- keep `directDist` fixed during optimization

## Included Sample Maps

Curated PT/NI map files for both worked datasets are in `sample_maps/`.
