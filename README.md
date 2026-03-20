# XRD Calibration Suite

This repository packages three XRD tools:

1. `part1_map_viewer`: interactive map click viewer (left: metric map, right: spectrum or image)
2. `part2_annulus_mask`: interactive annulus/phi mask builder (Dioptas-compatible `.mask`)
3. `part3_poni_optimization`: PONI optimization workflow (grid + local refinement)

The code is oriented to your existing PIMEGA-style datasets and keeps the latest project conventions:
- `col=0` is excluded in PONI selection logic
- robust row-fit outlier filtering is enabled
- local refinement supports fast top-10 optimization with essential top-20 validation
- `directDist` is kept fixed during optimization

## Environment

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Part 1: Interactive Map Viewer

### GUI launcher (path picker, no typing)

```bash
python part1_map_viewer/launch_map_viewer_gui.py
```

### CLI

```bash
python part1_map_viewer/view_map_with_1d.py --help
```

## Part 2: Annulus Mask Builder

### GUI launcher (file dialogs for `.poni`, `.h5`, output)

```bash
python part2_annulus_mask/launch_annulus_builder_gui.py
```

### CLI

```bash
python part2_annulus_mask/interactive_annulus_mask_builder.py --help
```

## Part 3: PONI Optimization

Main scripts:
- `poni_theta_phi_check.py`: top-point selection + per-file theta0 + cake + theta-vs-phi
- `poni_grid_search_4d.py`: 4D Fit2D grid scan (`dx`, `dy`, `dtilt`, `dtiltplan`)
- `poni_local_refine.py`: coordinate-descent local refinement + essential top-20 validation

### Example: local refinement from a known good start

```bash
python part3_poni_optimization/poni_local_refine.py \
  --base-poni E:/XRD/proc/proc/LaB6_003_25keV_poni.poni \
  --root-dir E:/XRD/data/PIMEGA_gridscan_00_47_31_restored_files \
  --pt-map-path E:/XRD/data/PIMEGA_gridscan_00_47_31_restored_files/hdf5_images_output/maps/map_pt_roi_power.npy \
  --mask-source F:/NMR/NMR/py_projects/xrd/annulus_phi_mask.mask \
  --mask-transform flipud \
  --start-dx -1.6 --start-dy 0 --start-dtilt 0.22 --start-dtiltplan 1.0 \
  --top-n 10 --top-n-essential 20 --max-iters 12 \
  --out-dir E:/XRD/data/PIMEGA_gridscan_00_47_31_restored_files/hdf5_images_output/poni_local_refine
```

### Example: coarse 4D grid

```bash
python part3_poni_optimization/poni_grid_search_4d.py \
  --base-poni E:/XRD/proc/proc/LaB6_003_25keV_poni.poni \
  --root-dir E:/XRD/data/PIMEGA_gridscan_00_47_31_restored_files \
  --pt-map-path E:/XRD/data/PIMEGA_gridscan_00_47_31_restored_files/hdf5_images_output/maps/map_pt_roi_power.npy \
  --mask-source F:/NMR/NMR/py_projects/xrd/annulus_phi_mask.mask \
  --mask-transform flipud \
  --top-n 10 \
  --dx-values=-1,0,1 --dy-values=-1,0,1 --dtilt-values=-0.1,0,0.1 --dtiltplan-values=-6,0,6 \
  --out-dir E:/XRD/data/PIMEGA_gridscan_00_47_31_restored_files/hdf5_images_output/poni_grid_search_4d
```

## Notes

- Paths in examples use Windows format; adjust as needed.
- Curated sample PT/NI maps for both worked datasets are included under `sample_maps/`.
