#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
handeye_calibrate.py - Hand-Eye Calibration Computation Module

Responsibilities:
- Load collected calibration data from disk
- Perform calibration computation (NO hardware interaction)
- Save calibration results

Design: Pure Computation
- This module ONLY handles calibration computation
- No hardware dependencies (robot or camera)
- Can be run offline with historical data
"""

import os
import sys
import json
import numpy as np
import cv2
from datetime import datetime

# Import calibration modules
from calibration_common import (
    prepare_calibration_data,
    CameraIntrinsicsManager
)
from improve_data_quality import DataQualityFilter, RANSACFilter
from calibration_algorithms import HandEyeCalibration, ErrorAnalyzer


class HandEyeCalibrator:
    """Hand-eye calibration computer

    Pure computation - no hardware dependencies
    """

    VERSION = "4.0.0"  # 新增内参标定和内外参联合优化

    def __init__(self, config=None):
        """Initialize calibrator

        Args:
            config: Optional, configuration dict
        """
        print(f"HandEyeCalibrator v{self.VERSION}")
        print("="*60)

        # Directory configuration
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_data_dir = os.path.join(self.script_dir, "calibration_data")
        self.verified_data_dir = os.path.join(self.script_dir, "verified_data")
        self.config_dir = os.path.join(self.script_dir, "config")  # Fixed: removed dirname

        # Load configuration from file
        config_file = os.path.join(self.config_dir, "calibration_config.yaml")
        if os.path.exists(config_file):
            import yaml
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
        else:
            file_config = {}

        # Default configuration
        self.config = {
            'mode': file_config.get('calibration_mode', 'eye_in_hand'),
            'board_size': tuple(file_config.get('chessboard', {}).get('board_size', [6, 4])),
            'chessboard_size_mm': file_config.get('chessboard', {}).get('square_size_mm', 50.0),
            'quality_filter': file_config.get('quality_filter', {
                'min_motion_mm': 5.0,
                'min_rotation_deg': 2.0,
                'boundary_margin_ratio': 0.10,
                'max_pitch_deg': 70.0
            }),
            'ransac': file_config.get('ransac', {
                'threshold_mm': 6.0,
                'min_inlier_ratio': 0.3
            }),
            'min_frames': file_config.get('calibration', {}).get('min_frames', 10)
        }

        # Override with user config
        if config:
            self.config.update(config)

        # Unpack common config
        self.board_size = tuple(self.config['board_size'])
        self.chessboard_size_mm = self.config['chessboard_size_mm']

    # ========================================================================
    # Camera Intrinsics Calibration
    # ========================================================================

    def calibrate_camera_intrinsics(self, collected_data, image_size=(1280, 720)):
        """Calibrate camera intrinsics using chessboard images

        Args:
            collected_data: List of collected calibration data with corners
            image_size: (width, height) of images

        Returns:
            tuple: (camera_matrix, dist_coeffs, reprojection_error) or (None, None, None)
        """
        print("\n" + "="*60)
        print("相机内参标定")
        print("="*60)

        # Prepare object points (3D points in real world space)
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objp *= (self.chessboard_size_mm / 1000.0)  # Convert to meters

        # Collect object points and image points from all frames
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane

        valid_frames = 0
        for data in collected_data:
            if 'corners' in data and data.get('chessboard_detected', True):
                objpoints.append(objp)
                imgpoints.append(data['corners'])
                valid_frames += 1

        if valid_frames < 10:
            print(f"❌ 错误: 有效帧数不足 ({valid_frames} < 10)")
            return None, None, None

        print(f"  使用 {valid_frames} 帧图像进行内参标定")

        # Calibrate camera
        print("  执行标定...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None,
            flags=cv2.CALIB_RATIONAL_MODEL  # Use rational distortion model
        )

        if not ret:
            print("❌ 内参标定失败")
            return None, None, None

        # Calculate reprojection error
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                             camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        mean_error = total_error / len(objpoints)

        print(f"\n📊 内参标定结果:")
        print(f"  焦距: fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
        print(f"  主点: cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
        print(f"  畸变系数: {dist_coeffs.flatten()}")
        print(f"  平均重投影误差: {mean_error:.4f} 像素")

        # Quality assessment
        if mean_error < 0.5:
            print(f"  质量评估: 🌟 优秀 (< 0.5px)")
        elif mean_error < 1.0:
            print(f"  质量评估: 👍 良好 (< 1.0px)")
        elif mean_error < 2.0:
            print(f"  质量评估: ⚠️  可接受 (< 2.0px)")
        else:
            print(f"  质量评估: ❌ 需要改进 (≥ 2.0px)")

        return camera_matrix, dist_coeffs, mean_error

    # ========================================================================
    # Load Data
    # ========================================================================

    def load_calibration_data(self, data_dir, redetect_corners=False):
        """Load calibration data from directory

        Args:
            data_dir: Directory containing calibration data
            redetect_corners: If True, re-detect corners from images instead of using saved corners

        Supports both new format (calibration_data.json) and legacy format (poses.txt + images)

        Args:
            data_dir: Data directory path

        Returns:
            tuple: (collected_data, camera_matrix, dist_coeffs, source, metadata)
                   or (None, None, None, None, None) if failed
        """
        print(f"\nLoading calibration data from: {data_dir}")

        # Try to load metadata (contains calibration mode info)
        metadata = {}
        metadata_file = os.path.join(data_dir, "calibration_metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if 'calibration_mode' in metadata:
                        print(f"  检测到标定模式: {metadata['calibration_mode']}")
            except Exception as e:
                print(f"  Warning: Failed to load metadata: {e}")

        # Try new format first (calibration_data.json)
        data_file = os.path.join(data_dir, "calibration_data.json")
        poses_file = os.path.join(data_dir, "poses.txt")

        collected_data = []

        if os.path.exists(data_file):
            # New format: calibration_data.json
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)

                # Load camera intrinsics if redetecting corners
                if redetect_corners:
                    print("🔄 重新检测角点模式")
                    camera_matrix, dist_coeffs, source = \
                        CameraIntrinsicsManager.get_camera_intrinsics(
                            config_dir=self.config_dir
                        )

                    if camera_matrix is None:
                        print(f"❌ 错误: 无法加载相机内参")
                        return None, None, None, None, None

                    print(f"📷 内参来源: {source}")

                # Process each frame
                for item in data:
                    # Skip frames where chessboard was not detected (if not redetecting)
                    if not redetect_corners and not item.get('chessboard_detected', True):
                        continue

                    frame_id = item['frame_id']

                    # Re-detect corners from image if requested
                    if redetect_corners:
                        image_file = os.path.join(data_dir, f"{frame_id}_Color.png")

                        if not os.path.exists(image_file):
                            print(f"⚠️  帧 {frame_id}: 找不到图像文件")
                            continue

                        # Read image
                        image = cv2.imread(image_file)
                        if image is None:
                            print(f"⚠️  帧 {frame_id}: 无法读取图像文件")
                            continue

                        # Detect chessboard corners
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        ret, corners = cv2.findChessboardCorners(
                            gray,
                            self.board_size,
                            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
                        )

                        if not ret:
                            print(f"⚠️  帧 {frame_id}: 棋盘格检测失败")
                            continue

                        # Refine corners
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                        # Calculate reprojection error
                        reprojection_error = self._calculate_reprojection_error(
                            corners, camera_matrix, dist_coeffs
                        )

                        collected_data.append({
                            'frame_id': frame_id,
                            'pose': item['pose'],
                            'corners': corners,
                            'reprojection_error_px': reprojection_error
                        })

                        print(f"✅ 帧 {frame_id}: 棋盘格重新检测成功 (重投影误差: {reprojection_error:.3f}px)")

                    else:
                        # Use saved corners
                        # Skip if corners key is missing (shouldn't happen with chessboard_detected=true)
                        if 'corners' not in item:
                            print(f"  Warning: Frame {item['frame_id']} marked as detected but missing corners")
                            continue

                        corners = np.array(item['corners'], dtype=np.float32)
                        collected_data.append({
                            'frame_id': item['frame_id'],
                            'pose': item['pose'],
                            'corners': corners
                        })

                if redetect_corners:
                    print(f"  重新检测: {len(collected_data)}/{len(data)} 帧成功")
                else:
                    print(f"  Loaded {len(collected_data)}/{len(data)} frames with valid chessboard detection")

            except Exception as e:
                print(f"  Error loading data: {e}")
                return None, None, None, None, None

        elif os.path.exists(poses_file):
            # Legacy format: poses.txt + image files
            print(f"  Using legacy format (poses.txt + images)")

            try:
                # Read poses
                poses = []
                with open(poses_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split()
                        if len(parts) >= 7:
                            frame_id = int(parts[0])
                            roll, pitch, yaw = float(parts[1]), float(parts[2]), float(parts[3])
                            x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                            poses.append({
                                'frame_id': frame_id,
                                'pose': [x, y, z, roll, pitch, yaw]
                            })

                print(f"📊 找到 {len(poses)} 个位姿数据")

                # Load camera intrinsics first (needed for corner detection)
                camera_matrix, dist_coeffs, source = \
                    CameraIntrinsicsManager.get_camera_intrinsics(
                        config_dir=self.config_dir
                    )

                if camera_matrix is None:
                    print(f"❌ 错误: 无法加载相机内参")
                    return None, None, None, None, None

                print(f"📷 内参来源: {source}")

                # Process each image to detect corners
                print("\n检测棋盘格...")
                for pose_data in poses:
                    frame_id = pose_data['frame_id']
                    image_file = os.path.join(data_dir, f"{frame_id}_Color.png")

                    if not os.path.exists(image_file):
                        print(f"⚠️  帧 {frame_id}: 找不到图像文件")
                        continue

                    # Read image
                    image = cv2.imread(image_file)
                    if image is None:
                        print(f"⚠️  帧 {frame_id}: 无法读取图像文件")
                        continue

                    # Detect chessboard corners
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(
                        gray,
                        self.board_size,
                        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
                    )

                    if not ret:
                        print(f"⚠️  帧 {frame_id}: 棋盘格检测失败")
                        continue

                    # Refine corners
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    # Calculate reprojection error
                    reprojection_error = self._calculate_reprojection_error(
                        corners, camera_matrix, dist_coeffs
                    )

                    collected_data.append({
                        'frame_id': frame_id,
                        'pose': pose_data['pose'],
                        'corners': corners,
                        'reprojection_error_px': reprojection_error
                    })

                    print(f"✅ 帧 {frame_id}: 棋盘格检测成功 (重投影误差: {reprojection_error:.3f}px)")

                print(f"\n📊 有效数据: {len(collected_data)}/{len(poses)} 组")

                if len(collected_data) == 0:
                    print(f"  Error: No valid frames found")
                    return None, None, None, None, None

                return collected_data, camera_matrix, dist_coeffs, source, metadata

            except Exception as e:
                print(f"  Error processing legacy format: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None, None, None
        else:
            print(f"  Error: No calibration data found (neither calibration_data.json nor poses.txt)")
            return None, None, None, None, None

        # Load camera intrinsics (for new format)
        camera_matrix, dist_coeffs, source = \
            CameraIntrinsicsManager.get_camera_intrinsics(
                config_dir=self.config_dir
            )

        if camera_matrix is None:
            print(f"  Error: Failed to load camera intrinsics")
            return None, None, None, None, None

        print(f"  Intrinsics loaded ({source})")

        return collected_data, camera_matrix, dist_coeffs, source, metadata

    def _calculate_reprojection_error(self, corners, camera_matrix, dist_coeffs):
        """计算棋盘格角点的重投影误差

        Args:
            corners: 检测到的角点 (Nx1x2)
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数

        Returns:
            float: 平均重投影误差（像素）
        """
        # 准备棋盘格3D点
        objpoints = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objpoints[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objpoints *= self.chessboard_size_mm / 1000.0  # 转为米

        # 使用PnP求解棋盘格位姿
        ret, rvec, tvec = cv2.solvePnP(
            objpoints, corners,
            camera_matrix, dist_coeffs
        )

        if not ret:
            return float('inf')  # PnP失败返回无穷大

        # 重投影3D点到图像平面
        projected_points, _ = cv2.projectPoints(
            objpoints, rvec, tvec,
            camera_matrix, dist_coeffs
        )

        # 计算平均重投影误差
        error = np.linalg.norm(
            corners.reshape(-1, 2) - projected_points.reshape(-1, 2),
            axis=1
        ).mean()

        return error

    # ========================================================================
    # Joint Intrinsic-Extrinsic Optimization
    # ========================================================================

    def joint_calibration_optimization(self, collected_data, camera_matrix_init, dist_coeffs_init,
                                      R_cam2gripper_init, t_cam2gripper_init,
                                      calibration_mode='eye_in_hand'):
        """Joint optimization of camera intrinsics and hand-eye extrinsics

        Uses bundle adjustment to simultaneously optimize:
        - Camera intrinsic parameters (focal length, principal point, distortion)
        - Hand-eye transformation (rotation and translation)

        Args:
            collected_data: List of calibration data frames
            camera_matrix_init: Initial camera matrix (3x3)
            dist_coeffs_init: Initial distortion coefficients
            R_cam2gripper_init: Initial camera-to-gripper rotation
            t_cam2gripper_init: Initial camera-to-gripper translation
            calibration_mode: 'eye_in_hand' or 'eye_to_hand'

        Returns:
            tuple: (camera_matrix_opt, dist_coeffs_opt, R_cam2gripper_opt, t_cam2gripper_opt, final_error)
        """
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation as R_scipy

        print("\n" + "="*60)
        print("内外参联合优化 (Bundle Adjustment)")
        print("="*60)

        # Prepare chessboard 3D points
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objp *= (self.chessboard_size_mm / 1000.0)

        # Extract valid frames with corners
        valid_frames = []
        for data in collected_data:
            if 'corners' in data and 'pose' in data:
                valid_frames.append(data)

        print(f"  使用 {len(valid_frames)} 帧进行联合优化")

        # Build parameter vector
        # [fx, fy, cx, cy, k1, k2, p1, p2, k3, r_vec(3), t_vec(3)]
        fx, fy = camera_matrix_init[0, 0], camera_matrix_init[1, 1]
        cx, cy = camera_matrix_init[0, 2], camera_matrix_init[1, 2]

        k1, k2, p1, p2, k3 = dist_coeffs_init.flatten()[:5] if len(dist_coeffs_init) >= 5 else (0, 0, 0, 0, 0)

        r_vec_init = R_scipy.from_matrix(R_cam2gripper_init).as_rotvec()
        t_vec_init = t_cam2gripper_init.flatten()

        # Initial parameter vector
        params_init = np.concatenate([
            [fx, fy, cx, cy],      # Intrinsics (4)
            [k1, k2, p1, p2, k3],  # Distortion (5)
            r_vec_init,            # Rotation vector (3)
            t_vec_init             # Translation vector (3)
        ])  # Total: 15 parameters

        print(f"  初始参数向量长度: {len(params_init)}")
        print(f"    内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        print(f"    畸变: k1={k1:.6f}, k2={k2:.6f}, p1={p1:.6f}, p2={p2:.6f}, k3={k3:.6f}")
        print(f"    外参旋转: {r_vec_init}")
        print(f"    外参平移: {t_vec_init}")

        def residual_function(params, valid_frames, objp, calibration_mode):
            """Compute reprojection residuals for all frames"""
            # Extract parameters
            fx, fy, cx, cy = params[0:4]
            k1, k2, p1, p2, k3 = params[4:9]
            r_vec_cam2gripper = params[9:12]
            t_vec_cam2gripper = params[12:15]

            # Build camera matrix and distortion
            camera_matrix = np.array([[fx, 0, cx],
                                     [0, fy, cy],
                                     [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

            # Convert rotation vector to matrix
            R_cam2gripper = R_scipy.from_rotvec(r_vec_cam2gripper).as_matrix()
            t_cam2gripper = t_vec_cam2gripper.reshape(3, 1)

            residuals = []

            for data in valid_frames:
                # Get robot pose
                x, y, z, roll, pitch, yaw = data['pose']
                R_gripper2base = R_scipy.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
                t_gripper2base = np.array([[x], [y], [z]])

                # Compute chessboard pose in camera frame
                if calibration_mode == 'eye_in_hand':
                    # T_board2cam = T_board2base * T_base2gripper * T_gripper2cam
                    # T_board2base is identity (board is at origin/base)
                    # T_base2gripper = inv(T_gripper2base)
                    # T_gripper2cam = inv(T_cam2gripper)
                    R_base2gripper = R_gripper2base.T
                    t_base2gripper = -R_base2gripper @ t_gripper2base

                    R_gripper2cam = R_cam2gripper.T
                    t_gripper2cam = -R_gripper2cam @ t_cam2gripper

                    R_board2cam = R_gripper2cam @ R_base2gripper
                    t_board2cam = R_gripper2cam @ t_base2gripper + t_gripper2cam
                else:  # eye_to_hand
                    # T_board2cam = T_board2gripper * T_gripper2base * T_base2cam
                    # T_base2cam = inv(T_cam2base)
                    R_cam2base = R_cam2gripper  # In eye-to-hand, we calibrate cam2base
                    t_cam2base = t_cam2gripper

                    R_base2cam = R_cam2base.T
                    t_base2cam = -R_base2cam @ t_cam2base

                    R_board2cam = R_base2cam @ R_gripper2base
                    t_board2cam = R_base2cam @ t_gripper2base + t_base2cam

                # Convert to rotation vector for cv2.projectPoints
                rvec_board2cam, _ = cv2.Rodrigues(R_board2cam)

                # Project 3D points to 2D
                projected_points, _ = cv2.projectPoints(
                    objp, rvec_board2cam, t_board2cam,
                    camera_matrix, dist_coeffs
                )

                # Compute residuals (observed - projected)
                observed_points = data['corners'].reshape(-1, 2)
                projected_points = projected_points.reshape(-1, 2)

                residual = (observed_points - projected_points).flatten()
                residuals.extend(residual)

            return np.array(residuals)

        # Run optimization
        print("\n  执行联合优化...")
        print(f"    优化算法: Levenberg-Marquardt")
        print(f"    观测数: {len(valid_frames) * len(objp) * 2} (每帧 {len(objp)} 个角点)")

        result = least_squares(
            residual_function,
            params_init,
            args=(valid_frames, objp, calibration_mode),
            method='lm',  # Levenberg-Marquardt
            verbose=2,
            max_nfev=200  # Maximum function evaluations
        )

        if not result.success:
            print(f"\n⚠️  优化未收敛: {result.message}")
        else:
            print(f"\n✅ 优化成功收敛")

        # Extract optimized parameters
        params_opt = result.x
        fx_opt, fy_opt, cx_opt, cy_opt = params_opt[0:4]
        k1_opt, k2_opt, p1_opt, p2_opt, k3_opt = params_opt[4:9]
        r_vec_opt = params_opt[9:12]
        t_vec_opt = params_opt[12:15]

        camera_matrix_opt = np.array([[fx_opt, 0, cx_opt],
                                      [0, fy_opt, cy_opt],
                                      [0, 0, 1]], dtype=np.float64)
        dist_coeffs_opt = np.array([k1_opt, k2_opt, p1_opt, p2_opt, k3_opt], dtype=np.float64)

        R_cam2gripper_opt = R_scipy.from_rotvec(r_vec_opt).as_matrix()
        t_cam2gripper_opt = t_vec_opt.reshape(3, 1)

        # Compute final reprojection error
        final_residuals = residual_function(params_opt, valid_frames, objp, calibration_mode)
        final_error = np.sqrt(np.mean(final_residuals**2))

        print(f"\n📊 联合优化结果:")
        print(f"  优化前重投影误差: {np.sqrt(np.mean(residual_function(params_init, valid_frames, objp, calibration_mode)**2)):.4f} px")
        print(f"  优化后重投影误差: {final_error:.4f} px")
        print(f"\n  优化后内参:")
        print(f"    焦距: fx={fx_opt:.2f} ({fx_opt-fx:+.2f}), fy={fy_opt:.2f} ({fy_opt-fy:+.2f})")
        print(f"    主点: cx={cx_opt:.2f} ({cx_opt-cx:+.2f}), cy={cy_opt:.2f} ({cy_opt-cy:+.2f})")
        print(f"  优化后畸变: k1={k1_opt:.6f}, k2={k2_opt:.6f}, p1={p1_opt:.6f}, p2={p2_opt:.6f}, k3={k3_opt:.6f}")
        print(f"  优化后外参平移: {t_vec_opt * 1000} mm")

        return camera_matrix_opt, dist_coeffs_opt, R_cam2gripper_opt, t_cam2gripper_opt, final_error

    # ========================================================================
    # Calibration Computation
    # ========================================================================

    def calibrate(self, data_dir, mode=None, save_results=True, redetect_corners=False,
                  calib_mode='extrinsic'):
        """Perform hand-eye calibration computation

        Args:
            data_dir: Data directory path
            mode: 'eye_in_hand' or 'eye_to_hand' (None = use config default)
            save_results: Whether to save results to data_dir
            redetect_corners: If True, re-detect corners from images instead of using saved corners
            calib_mode: Calibration mode - 'intrinsic', 'extrinsic', or 'joint'
                       - 'intrinsic': Only calibrate camera intrinsics
                       - 'extrinsic': Only calibrate hand-eye extrinsics (default)
                       - 'joint': Joint optimization of both intrinsics and extrinsics

        Returns:
            tuple: (R_result, t_result, quality_result) or (None, None, None)
                   For intrinsic mode: (camera_matrix, dist_coeffs, reprojection_error)
                   For extrinsic/joint mode: (R_result, t_result, quality_result)
        """
        # Step 0: Load data (包含 metadata)
        print("\nStep 0: Load calibration data")
        collected_data, camera_matrix, dist_coeffs, source, data_metadata = \
            self.load_calibration_data(data_dir, redetect_corners=redetect_corners)

        if collected_data is None:
            print("Failed to load data")
            return None, None, None

        # 确定标定模式: 优先级 mode参数 > metadata > config默认值
        if mode is not None:
            calibration_mode = mode
        elif 'calibration_mode' in data_metadata:
            calibration_mode = data_metadata['calibration_mode']
            print(f"  使用数据的标定模式: {calibration_mode}")
        else:
            calibration_mode = self.config.get('mode', 'eye_in_hand')
            print(f"  使用默认标定模式: {calibration_mode}")

        print("\n" + "="*60)
        print(f"Hand-Eye Calibration Computation ({calibration_mode.upper().replace('_', '-')})")
        print(f"Calibration Type: {calib_mode.upper()}")
        print("="*60)

        # ====================================================================
        # INTRINSIC-ONLY CALIBRATION MODE
        # ====================================================================
        if calib_mode == 'intrinsic':
            print("\n模式: 仅标定相机内参")
            camera_matrix_calib, dist_coeffs_calib, reproj_error = \
                self.calibrate_camera_intrinsics(collected_data)

            if camera_matrix_calib is None:
                return None, None, None

            # Save intrinsics if requested
            if save_results:
                print("\n保存内参标定结果...")
                intrinsics_file = os.path.join(data_dir, "camera_intrinsics_calibrated.yaml")
                CameraIntrinsicsManager.save_to_file(
                    camera_matrix_calib, dist_coeffs_calib, intrinsics_file,
                    intrinsics_info={'reprojection_error': reproj_error,
                                   'calibration_frames': len(collected_data)}
                )
                print(f"  内参已保存到: {intrinsics_file}")

            return camera_matrix_calib, dist_coeffs_calib, reproj_error

        # ====================================================================
        # EXTRINSIC-ONLY or JOINT CALIBRATION MODE
        # ====================================================================

        # Step 1: Data quality filtering
        print("\nStep 1: Data quality filtering")
        filtered_data, filter_report = DataQualityFilter.apply_all_filters(
            collected_data, config=self.config.get('quality_filter', {}), verbose=True
        )

        if len(filtered_data) < self.config['min_frames']:
            print(f"\nInsufficient data after filtering: {len(filtered_data)} < {self.config['min_frames']}")
            return None, None, None

        # Step 2: Data preparation
        print("\nStep 2: Data preparation")
        R_g, t_g, R_t, t_t, frame_ids, reprojection_errors = \
            prepare_calibration_data(
                filtered_data,
                camera_matrix,
                dist_coeffs,
                self.board_size,
                self.chessboard_size_mm,
                compute_reprojection_errors=True
            )

        if len(R_g) < self.config['min_frames']:
            print(f"\nInsufficient valid poses: {len(R_g)} < {self.config['min_frames']}")
            return None, None, None

        print(f"  {len(R_g)} valid poses")
        print(f"  Reprojection error: {np.mean(reprojection_errors):.2f}±{np.std(reprojection_errors):.2f}px")

        # Step 2.5: Filter high reprojection error frames
        print("\nStep 2.5: Filter high reprojection error frames")

        # 保存过滤前的数据用于对比
        original_frame_ids = frame_ids.copy()
        original_errors = reprojection_errors.copy()

        R_g, t_g, R_t, t_t, frame_ids, reprojection_errors, removed = \
            DataQualityFilter.filter_reprojection_error(
                R_g, t_g, R_t, t_t, frame_ids, reprojection_errors,
                max_error_px=2.0
            )

        # 逐帧打印分析结果
        print(f"  过滤阈值: 重投影误差 ≤ 2.0px")
        print(f"\n  逐帧分析:")
        for i, (fid, err) in enumerate(zip(original_frame_ids, original_errors)):
            if err <= 2.0:
                print(f"    ✅ 帧 {fid}: 重投影误差 {err:.3f}px (保留)")
            else:
                print(f"    ❌ 帧 {fid}: 重投影误差 {err:.3f}px (移除)")

        # 统计总结
        print(f"\n  📊 过滤结果:")
        print(f"    原始帧数: {len(original_frame_ids)}")
        print(f"    保留帧数: {len(frame_ids)}")
        print(f"    移除帧数: {len(removed)}")
        if removed:
            removed_ids = [item['frame_id'] for item in removed]
            print(f"    移除的帧: {removed_ids}")

        if len(R_g) < self.config['min_frames']:
            print(f"\nInsufficient data after reprojection filtering: {len(R_g)} < {self.config['min_frames']}")
            return None, None, None

        # Step 3: RANSAC filtering
        print("\nStep 3: RANSAC geometric consistency filtering")
        print(f"  阈值: {self.config['ransac']['threshold_mm']}mm")
        best_inliers = RANSACFilter.ransac_filter_handeye(
            R_g, t_g, R_t, t_t, frame_ids,
            threshold=self.config['ransac']['threshold_mm'],
            min_inlier_ratio=self.config['ransac']['min_inlier_ratio']
        )

        if len(best_inliers) < self.config['min_frames']:
            print(f"\nInsufficient inliers after RANSAC: {len(best_inliers)} < {self.config['min_frames']}")
            return None, None, None

        # 逐帧打印 RANSAC 结果
        print(f"\n  逐帧分析:")
        inlier_set = set(best_inliers)
        for i, fid in enumerate(frame_ids):
            if i in inlier_set:
                print(f"    ✅ 帧 {fid}: 几何一致性良好 (inlier)")
            else:
                print(f"    ❌ 帧 {fid}: 几何异常 (outlier)")

        # 统计总结
        print(f"\n  📊 RANSAC 结果:")
        print(f"    原始帧数: {len(frame_ids)}")
        print(f"    Inliers: {len(best_inliers)}")
        print(f"    Outliers: {len(frame_ids) - len(best_inliers)}")
        print(f"    Inlier比例: {len(best_inliers)/len(frame_ids)*100:.1f}%")
        print(f"    保留的帧: {[frame_ids[i] for i in best_inliers]}")

        # Extract inliers
        R_g_inliers = [R_g[i] for i in best_inliers]
        t_g_inliers = [t_g[i] for i in best_inliers]
        R_t_inliers = [R_t[i] for i in best_inliers]
        t_t_inliers = [t_t[i] for i in best_inliers]
        inlier_frame_ids = [frame_ids[i] for i in best_inliers]

        # Step 4: Multi-algorithm fusion
        print("\nStep 4: Multi-algorithm fusion")
        R_result, t_result, best_method = HandEyeCalibration.multi_algorithm_fusion_with_mode(
            R_g_inliers, t_g_inliers, R_t_inliers, t_t_inliers,
            mode=calibration_mode,  # 传入标定模式
            verbose=True
        )

        if R_result is None:
            print("\nAll algorithms failed")
            return None, None, None

        print(f"  Best algorithm: {best_method}")

        # Step 5: Levenberg-Marquardt optimization
        print("\nStep 5: Levenberg-Marquardt non-linear optimization")
        R_optimized, t_optimized = HandEyeCalibration.levenberg_marquardt_optimization_with_mode(
            R_result, t_result, R_g_inliers, t_g_inliers,
            R_t_inliers, t_t_inliers,
            mode=calibration_mode,  # 传入标定模式
            verbose=True
        )

        # Step 5.5: Joint intrinsic-extrinsic optimization (if enabled)
        if calib_mode == 'joint':
            print("\nStep 5.5: 内外参联合优化")
            print("  使用外参初值进行联合Bundle Adjustment...")

            # Get inlier data with corners for joint optimization
            inlier_data = [filtered_data[i] for i in best_inliers]

            camera_matrix_joint, dist_coeffs_joint, R_joint, t_joint, joint_error = \
                self.joint_calibration_optimization(
                    inlier_data,
                    camera_matrix, dist_coeffs,
                    R_optimized, t_optimized,
                    calibration_mode
                )

            if camera_matrix_joint is not None:
                print(f"\n✅ 联合优化成功")
                print(f"  使用联合优化的内参和外参结果")

                # Update with joint optimization results
                camera_matrix = camera_matrix_joint
                dist_coeffs = dist_coeffs_joint
                R_optimized = R_joint
                t_optimized = t_joint

                # Save optimized intrinsics
                if save_results:
                    intrinsics_file = os.path.join(data_dir, "camera_intrinsics_joint_optimized.yaml")
                    CameraIntrinsicsManager.save_to_file(
                        camera_matrix_joint, dist_coeffs_joint, intrinsics_file,
                        intrinsics_info={'joint_optimization_error': joint_error,
                                       'calibration_frames': len(inlier_data)}
                    )
                    print(f"  联合优化后的内参已保存到: {intrinsics_file}")
            else:
                print(f"⚠️  联合优化失败，使用外参优化结果")

        # Step 6: Evaluation
        print("\nStep 6: Calibration result evaluation")
        quality_result = HandEyeCalibration.evaluate_calibration(
            R_optimized, t_optimized, R_g_inliers, t_g_inliers,
            R_t_inliers, t_t_inliers, verbose=True
        )

        # Print detailed calibration result (Legacy-style format)
        from scipy.spatial.transform import Rotation as R_scipy
        rotation = R_scipy.from_matrix(R_optimized)
        quaternion = rotation.as_quat()  # [x, y, z, w]
        euler_angles = rotation.as_euler('xyz', degrees=True)
        t_mm = t_optimized.flatten() * 1000.0

        print("\n" + "="*60)
        print("📊 最终标定结果:")
        print("="*60)
        print(f"  平移向量 (mm): X={t_mm[0]:.2f}, Y={t_mm[1]:.2f}, Z={t_mm[2]:.2f}")
        print(f"  欧拉角 (度):    Rx={euler_angles[0]:.2f}°, Ry={euler_angles[1]:.2f}°, Rz={euler_angles[2]:.2f}°")
        print(f"  四元数 (xyzw):  x={quaternion[0]:.4f}, y={quaternion[1]:.4f}, z={quaternion[2]:.4f}, w={quaternion[3]:.4f}")

        print(f"\n📈 标定质量:")
        avg_t = quality_result['translation_error_mm']['mean']
        avg_r = quality_result['rotation_error_deg']['mean']
        max_t = quality_result['translation_error_mm']['max']
        max_r = quality_result['rotation_error_deg']['max']

        print(f"  旋转重复性: 平均 {avg_r:.3f}° (最大 {max_r:.3f}°)")
        print(f"  平移重复性: 平均 {avg_t:.2f}mm (最大 {max_t:.2f}mm)")

        # Quality assessment
        if avg_t < 2.0 and avg_r < 0.3:
            quality_emoji = "🌟 优秀"
        elif avg_t < 5.0 and avg_r < 0.5:
            quality_emoji = "👍 良好"
        elif avg_t < 10.0 and avg_r < 1.0:
            quality_emoji = "⚠️  可接受"
        else:
            quality_emoji = "❌ 需要改进"

        print(f"  总体质量: {quality_emoji}")

        print(f"\n  使用数据: {len(best_inliers)}/{len(collected_data)} 帧")
        print(f"  算法: Optimized_{best_method}_with_RANSAC")

        # Step 6.5: Detailed per-frame quality analysis
        print("\n" + "="*60)
        print("📊 最终标定质量分析")
        print("="*60)

        # Get original poses for deviation calculation (from inlier data only)
        # Note: collected_data contains all original frames with pose info
        inlier_original_data = [collected_data[i] for i in best_inliers]

        # Initial reference position (assuming first pose in dataset)
        if inlier_original_data and 'pose' in inlier_original_data[0]:
            ref_pose = inlier_original_data[0]['pose']  # [x, y, z, roll, pitch, yaw] in meters and radians
            ref_pos_mm = np.array(ref_pose[:3]) * 1000.0  # Convert to mm
            ref_rpy_deg = np.rad2deg(ref_pose[3:])  # Convert to degrees
        else:
            # Fallback to default initial position if not available
            ref_pos_mm = np.array([300.0, 0.0, 300.0])  # Default mm
            ref_rpy_deg = np.array([180.0, 60.0, 180.0])  # Default degrees

        # Calculate per-frame errors
        for i, frame_id in enumerate(inlier_frame_ids):
            # Predict target pose using calibration result: R_t_pred = R_g @ R_c2g @ R_t2c
            R_pred = R_g_inliers[i] @ R_optimized @ R_t_inliers[i]
            t_pred = R_g_inliers[i] @ (R_optimized @ t_t_inliers[i] + t_optimized) + t_g_inliers[i]

            # Calculate pose deviation (if original pose available)
            pose_deviation_str = ""
            if i < len(inlier_original_data) and 'pose' in inlier_original_data[i]:
                pose = inlier_original_data[i]['pose']  # [x, y, z, roll, pitch, yaw] in meters and radians

                # Convert to mm and degrees
                pos_mm = np.array(pose[:3]) * 1000.0
                rpy_deg = np.rad2deg(pose[3:])

                # Calculate deviation from reference
                pos_dev = pos_mm - ref_pos_mm

                # Angle difference handling wrap-around (-180 to +180)
                def angle_diff(a, b):
                    diff = a - b
                    while diff > 180:
                        diff -= 360
                    while diff < -180:
                        diff += 360
                    return diff

                rpy_dev = np.array([
                    angle_diff(rpy_deg[0], ref_rpy_deg[0]),  # roll
                    angle_diff(rpy_deg[1], ref_rpy_deg[1]),  # pitch
                    angle_diff(rpy_deg[2], ref_rpy_deg[2])   # yaw
                ])

                pose_deviation_str = (
                    f" | 位姿偏差: "
                    f"ΔX={pos_dev[0]:+6.1f} ΔY={pos_dev[1]:+6.1f} ΔZ={pos_dev[2]:+6.1f}mm, "
                    f"ΔR={rpy_dev[0]:+5.1f} ΔP={rpy_dev[1]:+5.1f} ΔY={rpy_dev[2]:+5.1f}°"
                )

            # Calculate reprojection consistency error (compare with first frame)
            if i == 0:
                R_ref = R_pred
                t_ref = t_pred
                print(f"✅ 帧 {frame_id:2d}: 旋转误差  0.000°  平移误差   0.000mm{pose_deviation_str}")
            else:
                # Rotation error
                R_error = R_ref.T @ R_pred
                angle_error = np.degrees(np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1)))

                # Translation error
                t_error = np.linalg.norm(t_pred - t_ref) * 1000.0  # Convert to mm

                # Status indicator based on error magnitude
                if t_error < 3.0:
                    status = "✅"
                elif t_error < 5.0:
                    status = "⚠️"
                else:
                    status = "❌"

                print(f"{status} 帧 {frame_id:2d}: 旋转误差 {angle_error:6.3f}°  平移误差 {t_error:7.3f}mm{pose_deviation_str}")

        # Step 7: Save results
        if save_results:
            print("\nStep 7: Save calibration results")
            self._save_calibration_result(
                R_optimized, t_optimized, data_dir,
                {
                    'mode': calibration_mode,  # 保存标定模式
                    'method': f'Optimized_{best_method}_with_RANSAC',
                    'data_used': f'{len(best_inliers)}/{len(collected_data)}',
                    'quality': quality_result,
                    'inlier_frame_ids': inlier_frame_ids,
                    'filter_report': filter_report
                }
            )

        print("\n" + "="*60)
        print("Calibration completed successfully!")
        print("="*60)

        return R_optimized, t_optimized, quality_result

    def _save_calibration_result(self, R_cam2gripper, t_cam2gripper, save_dir, metadata):
        """Save calibration result to disk

        Args:
            R_cam2gripper: Camera to gripper rotation matrix
            t_cam2gripper: Camera to gripper translation vector
            save_dir: Save directory
            metadata: Metadata dict
        """
        from scipy.spatial.transform import Rotation as R
        import yaml

        # Convert to quaternion
        r = R.from_matrix(R_cam2gripper)
        quat = r.as_quat()  # [x, y, z, w]

        # 根据模式确定结果名称
        calibration_mode = metadata.get('mode', 'eye_in_hand')
        if calibration_mode == 'eye_to_hand':
            result_key = 'camera_to_base'
        else:
            result_key = 'camera_to_gripper'

        result = {
            'version': self.VERSION,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'mode': calibration_mode,
            'method': metadata.get('method', 'Unknown'),
            result_key: {
                'translation': {
                    'x': float(t_cam2gripper[0][0]),
                    'y': float(t_cam2gripper[1][0]),
                    'z': float(t_cam2gripper[2][0])
                },
                'rotation_matrix': R_cam2gripper.tolist(),
                'quaternion': {
                    'x': float(quat[0]),
                    'y': float(quat[1]),
                    'z': float(quat[2]),
                    'w': float(quat[3])
                }
            },
            'quality': metadata.get('quality', {}),
            'data_statistics': {
                'frames_used': metadata.get('data_used', 'N/A'),
                'inlier_frame_ids': metadata.get('inlier_frame_ids', [])
            }
        }

        # Generate timestamped filenames
        timestamp = result['timestamp']
        json_file = os.path.join(save_dir, f"hand_eye_calibration_result_{timestamp}.json")
        yaml_file = os.path.join(save_dir, f"hand_eye_calibration_result_{timestamp}.yaml")

        # Save JSON
        with open(json_file, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"  Results saved: {json_file}")

        # Save YAML (ROS friendly)
        with open(yaml_file, 'w') as f:
            yaml.dump(result, f)
        print(f"  Results saved: {yaml_file}")

        # Also save as latest (without timestamp) for easy access
        json_latest = os.path.join(save_dir, "hand_eye_calibration_result.json")
        yaml_latest = os.path.join(save_dir, "hand_eye_calibration_result.yaml")

        with open(json_latest, 'w') as f:
            json.dump(result, f, indent=4)

        with open(yaml_latest, 'w') as f:
            yaml.dump(result, f)

        print(f"  Latest results also saved to: hand_eye_calibration_result.json/yaml")

    # ========================================================================
    # Utility Functions
    # ========================================================================

    def list_calibration_data(self):
        """List available calibration data directories

        Returns:
            list: List of data directory paths
        """
        print("\nAvailable calibration data:")

        data_dirs = []
        # 支持新格式（带模式）和旧格式（不带模式）
        valid_prefixes = (
            "manual_calibration_eyeinhand_", "manual_calibration_eyetohand_",
            "replay_calibration_eyeinhand_", "replay_calibration_eyetohand_",
            "manual_calibration_", "replay_calibration_"  # 兼容旧格式
        )

        # Search calibration_data directory
        if os.path.exists(self.calibration_data_dir):
            for item in os.listdir(self.calibration_data_dir):
                if item.startswith(valid_prefixes):
                    full_path = os.path.join(self.calibration_data_dir, item)
                    if os.path.isdir(full_path):
                        data_dirs.append(full_path)

        # Search verified_data directory
        if os.path.exists(self.verified_data_dir):
            for item in os.listdir(self.verified_data_dir):
                if item.startswith(valid_prefixes):
                    full_path = os.path.join(self.verified_data_dir, item)
                    if os.path.isdir(full_path):
                        data_dirs.append(full_path)

        # Sort by timestamp (extract from directory name)
        # Format: manual_calibration_YYYYMMDD_HHMMSS or replay_calibration_YYYYMMDD_HHMMSS
        def extract_timestamp(dir_path):
            """Extract timestamp from directory name for sorting"""
            dir_name = os.path.basename(dir_path)
            # Split by underscore and get last two parts (date and time)
            parts = dir_name.split('_')
            if len(parts) >= 3:
                # Get YYYYMMDD_HHMMSS
                timestamp_str = '_'.join(parts[-2:])
                return timestamp_str
            return dir_name  # Fallback to full name

        data_dirs = sorted(data_dirs, key=extract_timestamp, reverse=True)  # Newest first

        if not data_dirs:
            print("  (None)")
            return []

        for i, dir_path in enumerate(data_dirs, 1):
            dir_name = os.path.basename(dir_path)

            # Check files
            data_file = os.path.join(dir_path, "calibration_data.json")
            poses_file = os.path.join(dir_path, "poses.txt")
            result_file = os.path.join(dir_path, "hand_eye_calibration_result.json")

            try:
                # Count poses and images
                pose_count = 0
                image_count = 0

                # Try new format first (calibration_data.json)
                if os.path.exists(data_file):
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                    pose_count = len(data)
                    data_format = "New"
                # Fallback to legacy format (poses.txt)
                elif os.path.exists(poses_file):
                    with open(poses_file, 'r') as f:
                        lines = f.readlines()
                    # Count non-empty, non-comment lines
                    pose_count = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
                    data_format = "Legacy"
                else:
                    print(f"  {i}. {dir_name} (No calibration data)")
                    continue

                # Count images (*.png files)
                image_files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
                image_count = len(image_files)

                # Check calibration status
                status = "✅ Calibrated" if os.path.exists(result_file) else "📝 Uncalibrated"

                # Display detailed information in one line
                print(f"  {i}. {dir_name} - Poses: {pose_count} | Images: {image_count} | {status}")

            except Exception as e:
                print(f"  {i}. {dir_name} (Error: {e})")

        return data_dirs


# ============================================================================
# Main Function - Interactive Menu
# ============================================================================

def main():
    """Main function - Interactive calibration computation"""
    print("="*60)
    print("Hand-Eye Calibration Computer v4.0.0")
    print("="*60)
    print("This tool performs calibration computation from collected data")
    print("No hardware required - pure computation")
    print("\nSupported calibration modes:")
    print("  - Camera intrinsics calibration")
    print("  - Hand-eye extrinsics calibration")
    print("  - Joint intrinsic-extrinsic optimization (Bundle Adjustment)")
    print("="*60)

    calibrator = HandEyeCalibrator()

    # List available data
    data_dirs = calibrator.list_calibration_data()

    if not data_dirs:
        print("\nNo calibration data found")
        print("Please run handeye_data_collect.py first to collect data")
        return

    # Select data to calibrate
    print("\nSelect data to calibrate:")
    choice = input("Enter number (or 'all' to calibrate all uncalibrated data): ").strip()

    # 选择标定模式
    print("\n标定模式:")
    print("  1. Eye-in-Hand (相机在末端)")
    print("  2. Eye-to-Hand (相机固定)")
    print("  0. 自动检测（从数据目录的 metadata 读取）")
    mode_choice = input("选择模式 (0/1/2) [默认 0]: ").strip()

    if mode_choice == "1":
        calibration_mode_default = "eye_in_hand"
    elif mode_choice == "2":
        calibration_mode_default = "eye_to_hand"
    else:
        calibration_mode_default = None  # 自动检测

    # 选择是否重新检测角点
    print("\n角点检测:")
    print("  1. 使用已保存的角点（快速）")
    print("  2. 重新从图像检测角点（更精确，但较慢）")
    redetect_choice = input("选择方式 (1/2) [默认 1]: ").strip()

    redetect_corners = (redetect_choice == "2")

    # 选择标定类型
    print("\n标定类型:")
    print("  1. 仅外参标定 (使用已有内参)")
    print("  2. 仅内参标定 (相机内参标定)")
    print("  3. 内外参联合优化 (Bundle Adjustment)")
    calib_type_choice = input("选择类型 (1/2/3) [默认 1]: ").strip()

    if calib_type_choice == "2":
        calib_mode = "intrinsic"
        print("\n📷 模式: 仅标定相机内参")
    elif calib_type_choice == "3":
        calib_mode = "joint"
        print("\n🔗 模式: 内外参联合优化")
        print("   将同时优化相机内参和手眼外参")
    else:
        calib_mode = "extrinsic"
        print("\n🤖 模式: 仅标定手眼外参")

    if choice.lower() == 'all':
        # Calibrate all uncalibrated data
        uncalibrated = []
        for dir_path in data_dirs:
            result_file = os.path.join(dir_path, "hand_eye_calibration_result.json")
            if not os.path.exists(result_file):
                uncalibrated.append(dir_path)

        if not uncalibrated:
            print("\nAll data already calibrated")
            return

        print(f"\nFound {len(uncalibrated)} uncalibrated datasets")
        confirm = input("Proceed to calibrate all? (y/n): ").strip().lower()

        if confirm != 'y':
            print("Cancelled")
            return

        for dir_path in uncalibrated:
            print(f"\n{'='*60}")
            print(f"Calibrating: {os.path.basename(dir_path)}")
            print(f"{'='*60}")

            # 使用指定模式或自动检测（在 calibrate 函数内部）
            R, t, quality = calibrator.calibrate(
                dir_path,
                mode=calibration_mode_default,
                save_results=True,
                redetect_corners=redetect_corners,
                calib_mode=calib_mode
            )

            if R is not None:
                print(f"✅ Success")
            else:
                print(f"❌ Failed")

    else:
        # Calibrate selected data
        try:
            index = int(choice) - 1
            if 0 <= index < len(data_dirs):
                selected_dir = data_dirs[index]
                print(f"\nSelected: {os.path.basename(selected_dir)}")

                # 使用指定模式或自动检测（在 calibrate 函数内部）
                R, t, quality = calibrator.calibrate(
                    selected_dir,
                    mode=calibration_mode_default,
                    save_results=True,
                    redetect_corners=redetect_corners,
                    calib_mode=calib_mode
                )

                if R is not None:
                    print(f"\n✅ Calibration successful!")
                else:
                    print(f"\n❌ Calibration failed")
            else:
                print("Invalid selection")
        except ValueError:
            print("Please enter a valid number or 'all'")


if __name__ == "__main__":
    main()
