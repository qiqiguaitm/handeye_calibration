#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration_algorithms.py - 手眼标定算法模块

提供:
- 多算法融合 (Tsai, Park, Horaud, Daniilidis, Andreff)
- 迭代非线性优化
- 标定结果评估
- 误差分析

Design: Linus "Good Taste" 原则
- 每个算法独立,可单独测试
- 失败时优雅降级
- 清晰的误差度量
"""

import numpy as np
import cv2
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


# ============================================================================
# 手眼标定算法
# ============================================================================

class HandEyeCalibration:
    """手眼标定算法集合"""

    @staticmethod
    def invert_transformations(R_list, t_list):
        """对变换列表求逆: T^(-1) = [R^T | -R^T*t]

        用于 eye-to-hand: 将 gripper2base 转换为 base2gripper

        Args:
            R_list: 旋转矩阵列表
            t_list: 平移向量列表

        Returns:
            (R_inv_list, t_inv_list): 逆变换列表
        """
        R_inv_list = []
        t_inv_list = []

        for R, t in zip(R_list, t_list):
            # T^(-1) = [R^T | -R^T * t]
            R_inv = R.T
            t_inv = -R_inv @ t

            R_inv_list.append(R_inv)
            t_inv_list.append(t_inv)

        return R_inv_list, t_inv_list

    @staticmethod
    def multi_algorithm_fusion(R_gripper2base, t_gripper2base,
                               R_target2cam, t_target2cam, verbose=True):
        """多算法融合 - 选择最佳算法

        测试5种经典手眼标定算法,选择位姿重复性最好的

        Args:
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            verbose: 是否显示详细信息

        Returns:
            tuple: (R_cam2gripper, t_cam2gripper, method_name) 或 (None, None, None)
        """
        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
            (cv2.CALIB_HAND_EYE_PARK, "Park"),
            (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
            (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
            (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff")
        ]

        best_result = None
        best_score = float('inf')
        best_method = None

        for method_id, method_name in methods:
            try:
                R_test, t_test = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=method_id
                )

                # 评估标定质量: 位姿重复性
                t_errors = []
                r_errors = []

                # 计算每帧下target在base坐标系的位姿
                for i in range(len(R_gripper2base)):
                    R_pred = R_gripper2base[i] @ R_test @ R_target2cam[i]
                    t_pred = R_gripper2base[i] @ (R_test @ t_target2cam[i] + t_test) + t_gripper2base[i]

                    if i == 0:
                        R_ref = R_pred
                        t_ref = t_pred
                    else:
                        # 平移误差 (mm)
                        t_error = np.linalg.norm(t_pred - t_ref) * 1000
                        t_errors.append(t_error)

                        # 旋转误差 (度)
                        R_error = R_ref.T @ R_pred
                        r_error = np.degrees(np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1)))
                        r_errors.append(r_error)

                avg_t_error = np.mean(t_errors) if t_errors else 0
                avg_r_error = np.mean(r_errors) if r_errors else 0

                # 综合评分: 归一化后加权求和
                # 平移: 5mm = 1.0, 旋转: 0.5° = 1.0
                score = (avg_t_error / 5.0) + (avg_r_error / 0.5)

                if verbose:
                    print(f"   {method_name}: 平移={avg_t_error:.3f}mm, 旋转={avg_r_error:.3f}°, 综合={score:.3f}")

                if score < best_score:
                    best_score = score
                    best_result = (R_test, t_test)
                    best_method = method_name

            except Exception as e:
                if verbose:
                    print(f"   {method_name}: 失败 ({e})")

        if best_result:
            return best_result[0], best_result[1], best_method

        return None, None, None

    @staticmethod
    def multi_algorithm_fusion_with_mode(R_gripper2base, t_gripper2base,
                                          R_target2cam, t_target2cam,
                                          mode='eye_in_hand', verbose=True):
        """多算法融合 - 支持 eye-in-hand 和 eye-to-hand

        Args:
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            mode: 'eye_in_hand' 或 'eye_to_hand'
            verbose: 是否显示详细信息

        Returns:
            tuple: (R_result, t_result, method_name) 或 (None, None, None)
        """
        if verbose:
            if mode == 'eye_in_hand':
                print(f"  标定模式: Eye-in-Hand (相机在末端)")
                print(f"  求解: camera_to_gripper")
            else:
                print(f"  标定模式: Eye-to-Hand (相机固定)")
                print(f"  求解: camera_to_base")

        # Eye-to-hand: 需要对机器人位姿求逆
        if mode == 'eye_to_hand':
            if verbose:
                print("  转换: gripper2base → base2gripper")
            R_A, t_A = HandEyeCalibration.invert_transformations(
                R_gripper2base, t_gripper2base
            )
        else:
            # Eye-in-hand: 直接使用
            R_A, t_A = R_gripper2base, t_gripper2base

        # 使用统一的算法（完全复用现有代码）
        R_result, t_result, method_name = HandEyeCalibration.multi_algorithm_fusion(
            R_A, t_A, R_target2cam, t_target2cam, verbose=verbose
        )

        return R_result, t_result, method_name

    @staticmethod
    def iterative_optimization(R_initial, t_initial, R_gripper2base,
                              t_gripper2base, R_target2cam, t_target2cam,
                              verbose=True):
        """迭代非线性优化

        使用L-BFGS-B优化器微调标定结果,最小化位姿重复性误差

        Args:
            R_initial: 初始旋转矩阵
            t_initial: 初始平移向量
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            verbose: 是否显示详细信息

        Returns:
            tuple: (R_optimized, t_optimized)
        """

        def objective(params):
            """目标函数: 最小化位姿重复性误差"""
            # 解析参数
            rvec = params[:3]
            tvec = params[3:6]

            # 转换为旋转矩阵
            R_opt, _ = cv2.Rodrigues(rvec)
            t_opt = tvec.reshape(3, 1)

            # 计算每帧下target在base坐标系的位姿
            R_preds = []
            t_preds = []

            for i in range(len(R_gripper2base)):
                R_pred = R_gripper2base[i] @ R_opt @ R_target2cam[i]
                t_pred = R_gripper2base[i] @ (R_opt @ t_target2cam[i] + t_opt) + t_gripper2base[i]
                R_preds.append(R_pred)
                t_preds.append(t_pred)

            # 以第一帧为参考,计算所有帧的偏差
            R_ref = R_preds[0]
            t_ref = t_preds[0]

            total_error = 0
            for i in range(1, len(R_preds)):
                # 旋转误差 (度)
                R_error = R_ref.T @ R_preds[i]
                angle_error = np.degrees(np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1)))

                # 平移误差 (mm)
                t_error = np.linalg.norm(t_preds[i] - t_ref) * 1000

                # 综合误差 (1° = 10mm)
                total_error += t_error * 0.8   + angle_error * 10.0  * 0.2

            return total_error

        # 初始参数
        rvec_init, _ = cv2.Rodrigues(R_initial)
        tvec_init = t_initial.flatten()
        params_init = np.concatenate([rvec_init.flatten(), tvec_init])

        # 计算初始误差
        error_before = objective(params_init)

        # 设置边界: 允许微调 (±5°, ±10mm)
        bounds = []
        for i in range(3):
            # 旋转: ±5° ≈ ±0.087 rad
            bounds.append((rvec_init[i] - 0.087, rvec_init[i] + 0.087))
        for i in range(3):
            # 平移: ±10mm = ±0.01m
            bounds.append((tvec_init[i] - 0.01, tvec_init[i] + 0.01))

        # L-BFGS-B优化
        result = minimize(
            objective,
            params_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 50,
                'ftol': 1e-6,
                'gtol': 1e-5
            }
        )

        # 解析结果
        rvec_opt = result.x[:3]
        tvec_opt = result.x[3:6]
        R_opt, _ = cv2.Rodrigues(rvec_opt)
        t_opt = tvec_opt.reshape(3, 1)

        # 计算优化后误差
        error_after = objective(result.x)

        # 如果优化后更差,退回初始值
        if error_after > error_before:
            if verbose:
                print(f"   ⚠️  优化未改善结果，使用初始值")
                print(f"   优化前: {error_before:.3f}, 优化后: {error_after:.3f}")
            return R_initial, t_initial
        else:
            improvement = (error_before - error_after) / error_before * 100
            if verbose:
                print(f"   ✅ 优化成功: {error_before:.3f} → {error_after:.3f} (改善{improvement:.1f}%)")
            return R_opt, t_opt

    @staticmethod
    def iterative_optimization_with_mode(R_initial, t_initial, R_gripper2base,
                                          t_gripper2base, R_target2cam, t_target2cam,
                                          mode='eye_in_hand', verbose=True):
        """迭代非线性优化 - 支持两种模式

        Args:
            R_initial: 初始旋转矩阵
            t_initial: 初始平移向量
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            mode: 'eye_in_hand' 或 'eye_to_hand'
            verbose: 是否显示详细信息

        Returns:
            tuple: (R_optimized, t_optimized)
        """
        # Eye-to-hand: 转换坐标
        if mode == 'eye_to_hand':
            R_A, t_A = HandEyeCalibration.invert_transformations(
                R_gripper2base, t_gripper2base
            )
        else:
            R_A, t_A = R_gripper2base, t_gripper2base

        # 复用现有优化算法
        return HandEyeCalibration.iterative_optimization(
            R_initial, t_initial, R_A, t_A,
            R_target2cam, t_target2cam, verbose=verbose
        )

    @staticmethod
    def levenberg_marquardt_optimization(R_initial, t_initial, R_gripper2base,
                                         t_gripper2base, R_target2cam, t_target2cam,
                                         verbose=True):
        """Levenberg-Marquardt 非线性优化

        使用 LM 算法优化标定结果，最小化位姿重复性残差

        Args:
            R_initial: 初始旋转矩阵
            t_initial: 初始平移向量
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            verbose: 是否显示详细信息

        Returns:
            tuple: (R_optimized, t_optimized)
        """
        from scipy.optimize import least_squares

        def residual_function(params):
            """残差函数: 计算所有帧的位姿偏差

            返回残差向量，LM 会最小化 sum(residuals^2)
            """
            # 解析参数
            rvec = params[:3]
            tvec = params[3:6]

            # 转换为旋转矩阵
            R_opt, _ = cv2.Rodrigues(rvec)
            t_opt = tvec.reshape(3, 1)

            # 计算每帧下 target 在 base 坐标系的位姿
            R_preds = []
            t_preds = []

            for i in range(len(R_gripper2base)):
                R_pred = R_gripper2base[i] @ R_opt @ R_target2cam[i]
                t_pred = R_gripper2base[i] @ (R_opt @ t_target2cam[i] + t_opt) + t_gripper2base[i]
                R_preds.append(R_pred)
                t_preds.append(t_pred)

            # 以第一帧为参考，计算所有帧的残差
            R_ref = R_preds[0]
            t_ref = t_preds[0]

            residuals = []
            for i in range(1, len(R_preds)):
                # 旋转误差 (度)
                R_error = R_ref.T @ R_preds[i]
                angle_error = np.degrees(np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1)))

                # 平移误差 (mm)
                t_error = np.linalg.norm(t_preds[i] - t_ref) * 1000

                # 添加到残差向量（归一化：旋转权重 0.2，平移权重 0.8）
                residuals.append(angle_error * 10.0 * 0.2)  # 旋转: 1° = 10mm
                residuals.append(t_error * 0.8)              # 平移: mm

            return np.array(residuals)

        # 初始参数
        rvec_init, _ = cv2.Rodrigues(R_initial)
        tvec_init = t_initial.flatten()
        params_init = np.concatenate([rvec_init.flatten(), tvec_init])

        # 计算初始残差
        residuals_before = residual_function(params_init)
        error_before = np.sum(residuals_before**2)

        # 设置边界: 允许微调 (±5°, ±10mm)
        lower_bounds = []
        upper_bounds = []
        for i in range(3):
            # 旋转: ±5° ≈ ±0.087 rad
            lower_bounds.append(rvec_init[i] - 0.087)
            upper_bounds.append(rvec_init[i] + 0.087)
        for i in range(3):
            # 平移: ±10mm = ±0.01m
            lower_bounds.append(tvec_init[i] - 0.01)
            upper_bounds.append(tvec_init[i] + 0.01)

        # Levenberg-Marquardt 优化
        result = least_squares(
            residual_function,
            params_init,
            method='lm',  # Levenberg-Marquardt
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=100,
            verbose=0
        )

        # 解析结果
        rvec_opt = result.x[:3]
        tvec_opt = result.x[3:6]
        R_opt, _ = cv2.Rodrigues(rvec_opt)
        t_opt = tvec_opt.reshape(3, 1)

        # 计算优化后残差
        residuals_after = residual_function(result.x)
        error_after = np.sum(residuals_after**2)

        # 如果优化后更差，退回初始值
        if error_after > error_before:
            if verbose:
                print(f"   ⚠️  LM优化未改善结果，使用初始值")
                print(f"   优化前: {error_before:.3f}, 优化后: {error_after:.3f}")
            return R_initial, t_initial
        else:
            improvement = (error_before - error_after) / error_before * 100
            if verbose:
                print(f"   ✅ LM优化成功: {error_before:.3f} → {error_after:.3f} (改善{improvement:.1f}%)")
                print(f"   迭代次数: {result.nfev}, 状态: {result.status} ({result.message})")
            return R_opt, t_opt

    @staticmethod
    def levenberg_marquardt_optimization_with_mode(R_initial, t_initial, R_gripper2base,
                                                    t_gripper2base, R_target2cam, t_target2cam,
                                                    mode='eye_in_hand', verbose=True):
        """Levenberg-Marquardt 优化 - 支持两种模式

        Args:
            R_initial: 初始旋转矩阵
            t_initial: 初始平移向量
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            mode: 'eye_in_hand' 或 'eye_to_hand'
            verbose: 是否显示详细信息

        Returns:
            tuple: (R_optimized, t_optimized)
        """
        # Eye-to-hand: 转换坐标
        if mode == 'eye_to_hand':
            R_A, t_A = HandEyeCalibration.invert_transformations(
                R_gripper2base, t_gripper2base
            )
        else:
            R_A, t_A = R_gripper2base, t_gripper2base

        # 调用 LM 优化算法
        return HandEyeCalibration.levenberg_marquardt_optimization(
            R_initial, t_initial, R_A, t_A,
            R_target2cam, t_target2cam, verbose=verbose
        )

    @staticmethod
    def evaluate_calibration(R_cam2gripper, t_cam2gripper, R_gripper2base,
                            t_gripper2base, R_target2cam, t_target2cam,
                            verbose=True):
        """评估标定结果质量

        计算位姿重复性误差和稳定性

        Args:
            R_cam2gripper: 相机到夹爪的旋转矩阵
            t_cam2gripper: 相机到夹爪的平移向量
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            verbose: 是否显示详细信息

        Returns:
            dict: 评估结果
        """
        # 计算每帧下target在base坐标系的位姿
        R_preds = []
        t_preds = []

        for i in range(len(R_gripper2base)):
            R_pred = R_gripper2base[i] @ R_cam2gripper @ R_target2cam[i]
            t_pred = R_gripper2base[i] @ (R_cam2gripper @ t_target2cam[i] + t_cam2gripper) + t_gripper2base[i]
            R_preds.append(R_pred)
            t_preds.append(t_pred)

        # 计算重复性误差
        R_ref = R_preds[0]
        t_ref = t_preds[0]

        t_errors = []
        r_errors = []

        for i in range(1, len(R_preds)):
            # 平移误差 (mm)
            t_error = np.linalg.norm(t_preds[i] - t_ref) * 1000
            t_errors.append(t_error)

            # 旋转误差 (度)
            R_error = R_ref.T @ R_preds[i]
            angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
            r_error = np.degrees(angle)
            r_errors.append(r_error)

        # 统计量
        result = {
            'translation_error_mm': {
                'mean': np.mean(t_errors),
                'std': np.std(t_errors),
                'max': np.max(t_errors),
                'median': np.median(t_errors)
            },
            'rotation_error_deg': {
                'mean': np.mean(r_errors),
                'std': np.std(r_errors),
                'max': np.max(r_errors),
                'median': np.median(r_errors)
            }
        }

        # 质量评级
        avg_t = result['translation_error_mm']['mean']
        avg_r = result['rotation_error_deg']['mean']

        if avg_t < 2.0 and avg_r < 0.3:
            quality = "优秀"
        elif avg_t < 5.0 and avg_r < 0.5:
            quality = "良好"
        elif avg_t < 10.0 and avg_r < 1.0:
            quality = "合格"
        else:
            quality = "较差"

        result['quality'] = quality

        if verbose:
            print(f"\n📊 标定质量评估:")
            print(f"  位姿重复性:")
            print(f"    平移: {avg_t:.2f}±{result['translation_error_mm']['std']:.2f}mm (max={result['translation_error_mm']['max']:.2f}mm)")
            print(f"    旋转: {avg_r:.3f}±{result['rotation_error_deg']['std']:.3f}° (max={result['rotation_error_deg']['max']:.3f}°)")
            print(f"  质量评级: {quality}")

        return result


# ============================================================================
# 误差分析工具
# ============================================================================

class ErrorAnalyzer:
    """误差分析工具"""

    @staticmethod
    def analyze_error_patterns(R_cam2gripper, t_cam2gripper, R_gripper2base,
                               t_gripper2base, R_target2cam, t_target2cam,
                               frame_ids):
        """分析误差模式,识别异常帧

        Args:
            R_cam2gripper: 相机到夹爪的旋转矩阵
            t_cam2gripper: 相机到夹爪的平移向量
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            frame_ids: 帧ID列表

        Returns:
            dict: 误差分析结果
        """
        # 计算每帧的位姿预测
        R_preds = []
        t_preds = []

        for i in range(len(R_gripper2base)):
            R_pred = R_gripper2base[i] @ R_cam2gripper @ R_target2cam[i]
            t_pred = R_gripper2base[i] @ (R_cam2gripper @ t_target2cam[i] + t_cam2gripper) + t_gripper2base[i]
            R_preds.append(R_pred)
            t_preds.append(t_pred)

        # 以第一帧为参考
        R_ref = R_preds[0]
        t_ref = t_preds[0]

        # 计算每帧误差
        frame_errors = []
        for i in range(1, len(R_preds)):
            t_error = np.linalg.norm(t_preds[i] - t_ref) * 1000
            R_error = R_ref.T @ R_preds[i]
            r_error = np.degrees(np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1)))

            frame_errors.append({
                'frame_id': frame_ids[i],
                'translation_error_mm': t_error,
                'rotation_error_deg': r_error,
                'total_error': t_error + r_error * 10.0  # 1° = 10mm
            })

        # 按误差排序
        frame_errors.sort(key=lambda x: x['total_error'], reverse=True)

        # 识别异常帧 (超过中位数 + 2倍MAD)
        errors = np.array([fe['total_error'] for fe in frame_errors])
        median = np.median(errors)
        mad = np.median(np.abs(errors - median))
        threshold = median + 2.0 * mad

        outliers = [fe for fe in frame_errors if fe['total_error'] > threshold]

        return {
            'frame_errors': frame_errors,
            'outliers': outliers,
            'outlier_threshold': threshold,
            'worst_frames': frame_errors[:5]  # 最差的5帧
        }

    @staticmethod
    def check_motion_diversity(R_gripper2base, t_gripper2base, verbose=True):
        """检查运动多样性

        Args:
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            verbose: 是否显示详细信息

        Returns:
            dict: 运动多样性分析结果
        """
        # 计算位移范围
        positions = np.array([t.flatten() for t in t_gripper2base])
        pos_range = positions.max(axis=0) - positions.min(axis=0)
        pos_range_mm = pos_range * 1000

        # 计算旋转范围
        euler_angles = []
        for R_mat in R_gripper2base:
            r = R.from_matrix(R_mat)
            euler = r.as_euler('xyz', degrees=True)
            euler_angles.append(euler)

        euler_angles = np.array(euler_angles)
        rot_range = euler_angles.max(axis=0) - euler_angles.min(axis=0)

        result = {
            'position_range_mm': pos_range_mm,
            'rotation_range_deg': rot_range,
            'sufficient_motion': (pos_range_mm > 50).all() and (rot_range > 10).all()
        }

        if verbose:
            print(f"\n🎯 运动多样性:")
            print(f"  位移范围: X={pos_range_mm[0]:.1f}mm, Y={pos_range_mm[1]:.1f}mm, Z={pos_range_mm[2]:.1f}mm")
            print(f"  旋转范围: Roll={rot_range[0]:.1f}°, Pitch={rot_range[1]:.1f}°, Yaw={rot_range[2]:.1f}°")
            print(f"  运动充分: {'✅' if result['sufficient_motion'] else '❌'}")

        return result
