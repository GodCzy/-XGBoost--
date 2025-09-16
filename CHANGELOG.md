# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-09-15
### Added
- Adaptive ensemble strategies with detailed metrics and SHAP aggregation.
- Rich Flask API endpoints exposing metadata, metrics, feature insights, optimization history, and monitor logs.
- Interactive dashboard displaying all backend capabilities including strategy selection and comparison.

### Changed
- Data preprocessing now normalizes column names and coerces numeric-like strings.
- Service training pipeline now auto-loads configured datasets and aggregates feature importances.

### Removed
- Obsolete experimental scripts and raw CSV sample replaced by integrated workflow.

## [0.1.1] - 2025-09-14
### Added
- Basic unit tests for preprocessing, model management, monitoring, and optimization.
- Testing instructions in README.
- Project changelog.

## [0.1.0] - 2025-09-14
### Added
- Initial version demonstrating emission prediction and optimization workflow.
