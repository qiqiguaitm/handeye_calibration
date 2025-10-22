# Release Notes

## v3.4.1 (2025-01-22) - Current Release

### 🐛 Bugfix Release

Critical fix for TCP offset functionality introduced in v3.4.0.

#### Fixed Issues
- **Error Code 9**: Fixed "not in correct mode" error when moving robot after TCP offset setting
  - Root cause: Setting TCP offset changes robot mode/state
  - Solution: Re-apply position control mode and sport state after TCP configuration
- Added explicit robot readiness confirmation

#### Affected Users
- All users of v3.4.0 experiencing motion errors after robot initialization
- Users with TCP offset configured in `calibration_config_xarm.yaml`

#### Upgrade from v3.4.0
Simply update to v3.4.1 - no configuration changes needed.

```bash
git checkout v3.4.1
```

---

## v3.4.0 (2025-01-22)

### 🎉 New Features: TCP Offset Support

#### What's New
Automatic TCP (Tool Center Point) offset configuration and setting for xArm robot to compensate for camera physical offset from the robot flange.

#### Key Features
1. **Configuration-based TCP Offset**
   - Define offset in `calibration_config_xarm.yaml`
   - Format: `[x, y, z, roll, pitch, yaw]` in mm and degrees
   - Default: `[0.0, 0.0, 172.0, 0.0, 0.0, 0.0]` (172mm Z-axis offset)

2. **Automatic Application**
   - TCP offset automatically set during robot initialization
   - Verification and confirmation built-in
   - Error handling and logging

3. **Testing Tools**
   - `test_tcp_offset.py`: Standalone test script
   - Verify configuration loading and offset setting
   - Confirm xArm API integration

4. **Documentation**
   - `TCP_OFFSET_README.md`: Complete usage guide
   - Configuration examples
   - API reference
   - Troubleshooting tips

#### Files Changed
- `config/calibration_config_xarm.yaml`: Added tcp_offset parameter
- `handeye_data_collect_xarm.py`:
  - TCP offset loading and setting
  - Robot initialization enhancement
- `test_tcp_offset.py`: New test script
- `TCP_OFFSET_README.md`: New documentation

#### Use Case
Essential for **Eye-in-Hand calibration** where camera is mounted on robot end-effector with physical offset from the flange center.

#### Example Configuration
```yaml
robot:
  tcp_offset: [0.0, 0.0, 172.0, 0.0, 0.0, 0.0]
```

#### Testing
- ✅ Tested on xArm 7 (IP: 192.168.1.236)
- ✅ Verified with 172mm Z-axis camera offset
- ✅ Confirmed offset persistence

---

## Version Comparison

| Version | TCP Offset | Error Code 9 Fix | Status |
|---------|-----------|------------------|---------|
| v3.4.1  | ✅        | ✅               | **Current** |
| v3.4.0  | ✅        | ❌               | Deprecated |
| v3.3.0  | ❌        | N/A              | Legacy |

---

## Migration Guide

### From v3.3.0 to v3.4.1

1. **Update Configuration**
   ```bash
   # Edit config/calibration_config_xarm.yaml
   # Add under robot section:
   robot:
     tcp_offset: [0.0, 0.0, 172.0, 0.0, 0.0, 0.0]
   ```

2. **Measure Your Offset**
   - Measure camera center position relative to flange center
   - Update tcp_offset values accordingly
   - Use calipers or measuring tools for accuracy

3. **Test the Setup**
   ```bash
   python3 test_tcp_offset.py
   ```

4. **Run Calibration**
   ```bash
   python3 handeye_data_collect_xarm.py
   ```

### From v3.4.0 to v3.4.1

No configuration changes needed. Just update the code:
```bash
git pull
git checkout v3.4.1
```

---

## Known Issues

None currently reported for v3.4.1.

---

## Compatibility

- **Python**: 3.7+
- **xArm SDK**: 1.15.3+
- **Robot**: xArm 5/6/7
- **Camera**: RealSense D435/D455
- **OS**: Linux (tested on Ubuntu 20.04)

---

## Getting Help

- Check `TCP_OFFSET_README.md` for detailed documentation
- Review `CHANGELOG.md` for all changes
- Test with `test_tcp_offset.py` before production use
- Report issues with detailed error logs

---

## Credits

Developed with assistance from Claude (Anthropic).

---

## Next Release

Planned features for v3.5.0:
- Camera intrinsics calibration
- Joint calibration optimization
- Enhanced error recovery
- Web-based monitoring interface
