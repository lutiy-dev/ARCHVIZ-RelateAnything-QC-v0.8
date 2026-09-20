# ARCHVIZ RelateAnything QC v0.8.2

Rejection-aware BEFORE vs AFTER geometry QC for AI-assisted architectural visualization in ComfyUI.

## v0.8.2
- normalized weighted cost (0..1 component penalties, divided by active weight sum)
- rejection-aware Hungarian assignment with dummy unmatched slots
- explicit `accept_match_cost` and `dummy_unmatched_cost`
- `Geometry = NOT_EVALUATED` when reliable matches are insufficient
- candidate/accepted cost diagnostics and reliable match fraction
- visual overlay remains GREEN matched, YELLOW drifted, RED missing, BLUE added
- RelateAnything semantic delta remains advisory only

## Defaults
`accept_match_cost=0.55`, `dummy_unmatched_cost=0.35`, `min_reliable_match_fraction=0.50`.

Update existing install with `git pull`, restart ComfyUI, then search for `RA · SMART BEFORE vs AFTER QC · v0.8.2`.
