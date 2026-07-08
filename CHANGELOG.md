# Changelog

All notable changes to this project are documented in this file.

## [1.2.1] - 2026-07-08

### Added

- Introduced pattern-based metric plotting integrated into the logging system.
- Added support for defining reusable metric plot patterns for cleaner, more consistent visual tracking.

### Changed

- Extended the logging architecture to support pattern-driven plotting workflows.
- Updated metric/logging flow to better accommodate structured visualization output.
- Refined logger integration to prepare for additional backend support beyond current options.

### Removed

- No explicit removals in this PR.

### Notes

- Current supported loggers remain:
	- basic
	- tensorboard
- This feature lays groundwork for future logger backends such as:
	- Weights & Biases (wandb)
	- ZanLM
	- ClearLM
	- MLflow
	- Neptune


## [1.2.0] - 2026-05-07

### Added

- Added a unified logger abstraction under `tipi/core/loggers/`.
- Added `BasicLogger` and `TensorBoardLogger` as first-class logger permanences.

### Changed

- Renamed `WandBManager` to `WandBLogger`.
- Refactored logger lifecycle to be backend-agnostic using `initialize()` instead of WandB-specific naming.
- Updated core runtime (`builder`, `executor`, `runner`) to use one unified `logger` permanence interface.

### Removed

- Removed legacy `WandBManager` compatibility alias.
- Removed legacy permanence key `wandb_logger`; use `logger`.
- Removed legacy helper logger and standalone helper surface that was no longer used in production.

### Notes

- This release contains breaking API changes for users importing or configuring WandB by old names.

## [1.1.0] - Previous

- Baseline for this changelog section.
