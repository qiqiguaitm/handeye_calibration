# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.4.0] - 2025-01-22

### Added
- **TCP Offset Support**: Automatic TCP offset configuration and setting for xArm robot
  - Configuration parameter in `calibration_config_xarm.yaml`
  - Automatic offset application during robot initialization
  - `set_tcp_offset()` function for programmatic offset setting
  - `test_tcp_offset.py` test script for verification
  - Complete documentation in `TCP_OFFSET_README.md`

### Changed
- Updated `handeye_data_collect_xarm.py` version to 3.4.0
- Enhanced robot initialization sequence with TCP offset setup
- Improved configuration loading with TCP offset validation
- Updated metadata version in collected calibration data

### Fixed
- Proper TCP offset handling for Eye-in-Hand calibration mode
- Camera-to-flange physical offset compensation

### Documentation
- Added `TCP_OFFSET_README.md` with comprehensive usage guide
- Documented xArm SDK API for TCP offset operations
- Added configuration examples and testing procedures

### Testing
- ✅ Tested with xArm robot (IP: 192.168.1.236)
- ✅ Verified 172mm Z-axis offset configuration
- ✅ Confirmed offset persistence and verification

---

## [3.3.0] - Previous Version

### Features
- Eye-in-Hand and Eye-to-Hand calibration modes
- Manual and replay trajectory collection
- Chessboard detection with corner refinement
- Camera intrinsics management
- RealSense camera support
- Multi-algorithm calibration fusion
- RANSAC-based outlier filtering
- Data quality assessment

### Platforms
- xArm robot support
- Piper robot support
- RealSense D435/D455 cameras

---

## Version Numbering

- **Major version**: Breaking changes or major feature additions
- **Minor version**: New features, backward compatible
- **Patch version**: Bug fixes and minor improvements

### Version Tags
- `v3.4.0` - TCP Offset Support
- `v3.3.0` - Dual Mode Calibration (Eye-in-Hand/Eye-to-Hand)
