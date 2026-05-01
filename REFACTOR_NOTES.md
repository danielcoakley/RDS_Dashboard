# Refactor Notes

## Client-specific assumptions found
- Dashboard title and guide text explicitly referenced **RDS Site**.
- Processing logic depended on fixed local file `seu_mapping.csv`.
- App expected one rigid input shape for energy data (`Period`, `Metered Sector`, `Utility`, date columns).
- Analysis defaults included hardcoded years in helper function signatures.

## Proposed replacements
- Added YAML client config layer under `config/clients/`.
- Added standard internal meter schema: `timestamp | meter_id | value | unit | commodity | source`.
- Added configurable mapping from uploaded columns to meter ids (`source_column` in config).
- Added reusable modules for config loading, standardisation, data quality, reporting, and EnPI scaffolding.

## Risky areas
- Legacy preprocessing and SEU workflows still depend on original file formats and mapping conventions.
- Existing baseline/regression logic still assumes weather files and meter-level compatibility.
- Streamlit app now reads energy upload twice (legacy + standardised path), which is safe but could be optimized.

## Functions/modules created
- `src/config_loader.py`: `load_client_config`, `validate_client_config`
- `src/data_standardisation.py`: `load_uploaded_data`, `standardise_meter_data`
- `src/data_quality.py`: `clean_meter_data`, `detect_data_gaps`, `detect_outliers`
- `src/reporting.py`: `calculate_consumption_summary`, `generate_iso_summary`
- `src/enpi_model.py`: `build_baseline_model`, `calculate_enpi`
- Added tests in `tests/` for config loading/validation and standardisation behaviors.

## SaaS platform planning update (2026-05-01)
- Added a dedicated SaaS transition plan with:
  - current-state assessment,
  - proposed target architecture,
  - prioritized epic-level issue backlog,
  - milestone plan from alignment through pilot readiness,
  - immediate next actions and risk mitigations.
- See `docs/SAAS_PLATFORM_PLAN.md`.
