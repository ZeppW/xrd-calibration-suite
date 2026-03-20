$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$builder = Join-Path $scriptDir "interactive_annulus_mask_builder.py"

$poni = "F:\NMR\NMR\py_projects\xrd\final_poni_share\PIMEGA_gridscan_00_47_31_final_best.poni"
$h5 = "E:\XRD\data\PIMEGA_gridscan_00_47_31_restored_files\PIMEGA_gridscan_00_47_31_429_restored.hdf5"
$outMask = Join-Path $scriptDir "demo_mask_00_47_31_429.mask"
$outSpec = Join-Path $scriptDir "demo_mask_00_47_31_429.json"

python $builder `
  --mode calibrated `
  --poni $poni `
  --h5 $h5 `
  --dataset data `
  --frame 0 `
  --out $outMask `
  --spec-out $outSpec

