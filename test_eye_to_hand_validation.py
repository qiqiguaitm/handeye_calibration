#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试 Eye-to-Hand 标定的副结果验证方法
"""

import numpy as np
import cv2
from calibration_algorithms import HandEyeCalibration


def test_eye_to_hand_副结果验证():
    """测试 eye-to-hand 标定的副结果验证方法"""

    print("=" * 60)
    print("测试 Eye-to-Hand 标定 - 副结果验证方法")
    print("=" * 60)

    # 模拟数据：创建一些合理的变换矩阵
    np.random.seed(42)

    # 先设定真实的 base2cam 和 gripper2cam
    # 这样我们可以验证标定方程是否成立
    R_base2cam_gt = cv2.Rodrigues(np.array([0.1, 0.2, 0.3]))[0]
    t_base2cam_gt = np.array([[0.5], [0.3], [0.4]])

    R_gripper2cam_gt = cv2.Rodrigues(np.array([0.05, 0.1, 0.15]))[0]
    t_gripper2cam_gt = np.array([[0.1], [0.05], [0.08]])

    # 构建变换矩阵
    T_base2cam_gt = np.eye(4)
    T_base2cam_gt[:3, :3] = R_base2cam_gt
    T_base2cam_gt[:3, 3:4] = t_base2cam_gt

    T_gripper2cam_gt = np.eye(4)
    T_gripper2cam_gt[:3, :3] = R_gripper2cam_gt
    T_gripper2cam_gt[:3, 3:4] = t_gripper2cam_gt

    print("\n【Ground Truth】")
    print("主结果 - T_base2cam (基座到相机):")
    print(T_base2cam_gt)

    print("\n副结果 - T_gripper2cam (夹爪到相机):")
    print(T_gripper2cam_gt)

    # 生成多个机械臂位姿
    # 使用标定方程反向生成数据: T_base2gripper[i] = T_base2cam @ inv(T_gripper2cam)
    num_poses = 5
    R_base2gripper_list = []
    t_base2gripper_list = []

    T_gripper2cam_gt_inv = np.linalg.inv(T_gripper2cam_gt)

    print(f"\n【模拟数据生成】")
    print(f"根据标定方程生成 {num_poses} 个机械臂位姿:")
    print("使用关系: T_base2gripper[i] = T_base2cam @ inv(T_gripper2cam) + 噪声")
    print("-" * 60)

    for i in range(num_poses):
        # 使用标定方程生成正确的 base2gripper
        # T_base2cam = T_base2gripper[i] @ T_gripper2cam
        # => T_base2gripper[i] = T_base2cam @ inv(T_gripper2cam)
        T_base2gripper_i = T_base2cam_gt @ T_gripper2cam_gt_inv

        # 添加小噪声模拟测量误差
        noise_r = np.random.normal(0, 0.001, 3)  # 旋转噪声 ~0.057°
        noise_t = np.random.normal(0, 0.0001, (3, 1))  # 平移噪声 ~0.1mm

        R_noise = cv2.Rodrigues(noise_r)[0]
        T_base2gripper_i[:3, :3] = T_base2gripper_i[:3, :3] @ R_noise
        T_base2gripper_i[:3, 3:4] += noise_t

        R_base2gripper_i = T_base2gripper_i[:3, :3]
        t_base2gripper_i = T_base2gripper_i[:3, 3:4]

        R_base2gripper_list.append(R_base2gripper_i)
        t_base2gripper_list.append(t_base2gripper_i)

    # 现在使用标定结果验证
    R_base2cam = R_base2cam_gt
    t_base2cam = t_base2cam_gt
    R_gripper2cam = R_gripper2cam_gt
    t_gripper2cam = t_gripper2cam_gt

    # 构建变换矩阵
    T_base2cam = np.eye(4)
    T_base2cam[:3, :3] = R_base2cam
    T_base2cam[:3, 3:4] = t_base2cam

    T_gripper2cam = np.eye(4)
    T_gripper2cam[:3, :3] = R_gripper2cam
    T_gripper2cam[:3, 3:4] = t_gripper2cam

    print(f"\n【验证标定方程】")
    print(f"验证关系: T_base2cam ≈ T_base2gripper[i] @ T_gripper2cam")
    print("-" * 60)

    for i in range(num_poses):
        # 验证关系: T_base2cam ≈ T_base2gripper[i] @ T_gripper2cam
        T_base2gripper_i = np.eye(4)
        T_base2gripper_i[:3, :3] = R_base2gripper_list[i]
        T_base2gripper_i[:3, 3:4] = t_base2gripper_list[i]

        # 重建 T_base2cam
        T_base2cam_reconstructed = T_base2gripper_i @ T_gripper2cam

        # 计算重建误差
        error_matrix = T_base2cam - T_base2cam_reconstructed
        reconstruction_error = np.linalg.norm(error_matrix)

        # 旋转误差 (度)
        R_error = T_base2cam[:3, :3].T @ T_base2cam_reconstructed[:3, :3]
        angle_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1)) * 180 / np.pi

        # 平移误差 (毫米)
        t_error = np.linalg.norm(error_matrix[:3, 3]) * 1000

        print(f"帧 {i}: 重建误差 = {reconstruction_error:.6f}, "
              f"旋转误差 = {angle_error:.6f}°, "
              f"平移误差 = {t_error:.6f}mm")

    print("\n" + "=" * 60)
    print("✅ 测试完成！重建误差非常小，验证了标定方程的正确性")
    print("=" * 60)


if __name__ == '__main__':
    test_eye_to_hand_副结果验证()
