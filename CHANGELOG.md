# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Basic directory organization
- Documentation setup
- Added LightGBM model training script with CPU support
- Added lightgbm dependency to environment.yaml

### Changed
- Removed GPU support from LightGBM model due to compatibility issues
- Updated model training scripts to handle date features correctly

### Fixed
- Fixed date conversion issues in feature preparation
- Fixed feature handling in model training scripts

## [0.1.0] - 2024-03-20

### Added
- Initial project setup
- Data processing pipeline
- Random Forest and XGBoost model implementations
- Documentation and README files 