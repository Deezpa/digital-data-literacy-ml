
# Data Guide (De-identified)

## Schema (Suggested)
- `participant_id` (hashed string)
- `age_band` (e.g., 18-25, 26-35, 36-45, 46+)
- `region` (state/zone code; avoid GPS)
- `literacy_baseline` (0–100)
- `module_hours` (float)
- `assessment_pre` (0–100)
- `assessment_post` (0–100)
- `improvement` (post - pre)
- `followup_90d` (0/1)
- `dropout_flag` (0/1)  **target**
- `device_access` (0/1)
- `net_availability` (0/1)
- `income_band` (ordinal code)

## Ethics & Privacy
- No direct identifiers (names, phone, exact addresses) in the repository.
- Use hashed IDs; store re-identification keys offline and never commit them.
- Aggregate or bucket sensitive attributes (e.g., age -> bands).
- Include consent language in your data collection workflows and document it externally.
- Consider group fairness reporting (demographic parity difference, TPR/FPR gaps).

## File Locations
- `data/raw/` : original CSV (de-identified)
- `data/processed/` : feature tables used by models
- `data/external/` : derived public/open datasets (if any)

