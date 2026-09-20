# ARCHVIZ RelateAnything QC v0.8

Smart BEFORE vs AFTER geometry QC for AI-assisted architectural visualization in ComfyUI.

## v0.8
- Hungarian assignment when SciPy is available
- row/column clustering
- normalized center + IoU + size + grid + neighborhood matching
- separate Detection and Geometry statuses
- visual overlay: GREEN matched, YELLOW drifted, RED missing, BLUE added
- RelateAnything semantic delta remains advisory only

## Install
Clone/copy this repository directly to:
`Q:\AI_ArchViz\ComfyUI_windows_portable\ComfyUI\custom_nodes\ARCHVIZ-RelateAnything-QC-v0.8`

Keep `ComfyUI-RelateAnything-ONNX-v06` installed. Restart ComfyUI and search for:
`RA · SMART BEFORE vs AFTER QC · v0.8`

Import `workflows/ARCHVIZ_RELATEANYTHING_BEFORE_AFTER_QC_v008.json`.

## Status logic
Detection PASS/WARN/FAIL is based on unmatched regions. Geometry PASS/WARN/FAIL is based on deterministic bbox/grid drift and order flips. RA relations never decide geometry truth.
