# Hand-Eye Calibration System

A robust hand-eye calibration system for robotic arms with RealSense cameras, supporting both Eye-in-Hand and Eye-to-Hand configurations.

## 🚀 Version: v3.4.1

Current stable release with TCP offset support and critical bugfixes.

## ✨ Features

- **Dual Calibration Modes**
  - Eye-in-Hand: Camera mounted on robot end-effector
  - Eye-to-Hand: Camera fixed, chessboard on robot

- **TCP Offset Support** (v3.4.0+)
  - Automatic configuration and application
  - Compensates for camera physical offset
  - Configurable via YAML

- **Robust Data Collection**
  - Manual collection mode
  - Trajectory replay mode
  - Real-time chessboard detection
  - Quality filtering and validation

- **Advanced Calibration**
  - Multi-algorithm fusion (TSAI, PARK, HORAUD, etc.)
  - RANSAC outlier filtering
  - Levenberg-Marquardt optimization
  - Comprehensive error analysis

- **Multi-Robot Support**
  - xArm 5/6/7
  - Piper robot
  - Easy to extend

## 📋 Quick Start

### 1. Installation

```bash
cd /home/shock/ztm/handeye_calibration
pip install -r requirements.txt  # If requirements file exists
```

### 2. Configuration

Edit `config/calibration_config_xarm.yaml`:

```yaml
# Calibration mode
calibration_mode: eye_in_hand  # or eye_to_hand

# Chessboard parameters
chessboard:
  board_size: [6, 4]
  square_size_mm: 50.0

# TCP offset (v3.4.0+)
robot:
  tcp_offset: [0.0, 0.0, 172.0, 0.0, 0.0, 0.0]
```

### 3. Data Collection

```bash
# For xArm
python3 handeye_data_collect_xarm.py

# For Piper
python3 handeye_data_collect_piper.py
```

Follow on-screen instructions:
- Press **S** to save current frame
- Press **ESC** to finish collection

### 4. Calibration

```bash
python3 handeye_calibrate.py
```

Select dataset and run calibration.

## 📁 Project Structure

```
handeye_calibration/
├── config/
│   ├── calibration_config_xarm.yaml    # xArm configuration
│   ├── calibration_config_piper.yaml   # Piper configuration
│   └── hand_camera_intrinsics_*.yaml   # Camera intrinsics
├── handeye_data_collect_xarm.py        # xArm data collection
├── handeye_data_collect_piper.py       # Piper data collection
├── handeye_calibrate.py                # Calibration computation
├── calibration_common.py               # Common utilities
├── calibration_algorithms.py           # Calibration algorithms
├── improve_data_quality.py             # Data quality filters
├── test_tcp_offset.py                  # TCP offset testing
├── CHANGELOG.md                        # Version history
├── RELEASE_NOTES.md                    # Release information
├── TCP_OFFSET_README.md                # TCP offset guide
└── README.md                           # This file
```

## 🧪 Testing

### Test TCP Offset (xArm only)
```bash
python3 test_tcp_offset.py
```

### Verify Configuration
```bash
# Load and verify config
python3 -c "
import yaml
with open('config/calibration_config_xarm.yaml') as f:
    config = yaml.safe_load(f)
    print('TCP Offset:', config['robot']['tcp_offset'])
"
```

## 📊 Calibration Quality

The system provides quality metrics:

| Metric | Excellent | Good | Acceptable |
|--------|-----------|------|------------|
| Translation Error | < 2mm | < 5mm | < 10mm |
| Rotation Error | < 0.3° | < 0.5° | < 1.0° |

## 🔧 Troubleshooting

### Error Code 9 (Not in correct mode)
**Fixed in v3.4.1**. If using v3.4.0, upgrade to v3.4.1:
```bash
git checkout v3.4.1
```

### Chessboard Not Detected
- Ensure proper lighting
- Check chessboard size matches configuration
- Verify camera focus
- Try different poses/angles

### Poor Calibration Quality
- Collect more diverse poses (20+ frames recommended)
- Ensure good chessboard visibility
- Check for motion blur
- Verify robot stability during capture

## 📚 Documentation

- **[RELEASE_NOTES.md](RELEASE_NOTES.md)**: Detailed release information
- **[CHANGELOG.md](CHANGELOG.md)**: Version history
- **[TCP_OFFSET_README.md](TCP_OFFSET_README.md)**: TCP offset guide

## 🏷️ Version Tags

```bash
# List all versions
git tag -l

# Checkout specific version
git checkout v3.4.1
```

Current tags:
- `v3.4.1` - Current (Bugfix for TCP offset)
- `v3.4.0` - TCP Offset Support (deprecated, use v3.4.1)
- `v3.3.0` - Dual Mode Support (legacy)

## 🤝 Contributing

1. Create feature branch from master
2. Make changes with clear commit messages
3. Test thoroughly
4. Update CHANGELOG.md
5. Submit for review

## 📝 License

[Specify your license here]

## 👥 Credits

Developed with assistance from Claude (Anthropic).

## 📧 Support

For issues and questions:
1. Check documentation
2. Review CHANGELOG and RELEASE_NOTES
3. Test with provided test scripts
4. Report with detailed logs

---

**Current Version**: v3.4.1 | **Last Updated**: 2025-01-22
