from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from typing import List, Tuple

import mujoco
import numpy as np


ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _load_pinocchio() -> object:
    try:
        return importlib.import_module("pinocchio")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少 pinocchio，请先安装 pinocchio。") from exc


def build_obs(
    data: mujoco.MjData,
    joint_count: int,
    box_body_id: int,
    ee_site_id: int,
    box_vel: np.ndarray,
    ee_vel: np.ndarray,
    obs_noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    qpos = data.qpos[:joint_count]
    qvel = data.qvel[:joint_count]
    box_pos = data.xpos[box_body_id].copy()
    ee_pos = data.site_xpos[ee_site_id].copy()
    rel = box_pos - ee_pos
    rel_vel = box_vel - ee_vel
    obs = np.concatenate([qpos, qvel, box_pos, ee_pos, rel, box_vel, ee_vel, rel_vel], dtype=np.float32)
    if obs_noise_std > 0.0:
        obs = obs + rng.normal(0.0, obs_noise_std, size=obs.shape).astype(np.float32)
    return obs


def _desired_target_pos(box_pos: np.ndarray, z_offset: float) -> np.ndarray:
    return np.array([box_pos[0], box_pos[1], box_pos[2] + z_offset], dtype=np.float64)


def _box_touch_target_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    box_body_id: int,
    box_geom_id: int,
    ee_world: np.ndarray,
    z_offset: float,
    touch_offset: float,
) -> np.ndarray:
    center = data.xpos[box_body_id].copy()
    center[2] += z_offset
    rot = data.xmat[box_body_id].reshape(3, 3)
    half = model.geom_size[box_geom_id, :3].astype(np.float64)

    rel_local = rot.T @ (ee_world - center)
    surf_local = np.clip(rel_local, -half, half)
    outside = np.any(np.abs(rel_local) > half)

    if not outside:
        # inside 时仅推到最近面；outside 时直接用最近点，避免面切换跳变
        margin = half - np.abs(rel_local)
        axis = int(np.argmin(margin))
        sgn = 1.0 if rel_local[axis] >= 0.0 else -1.0
        surf_local[axis] = sgn * half[axis]

    if touch_offset != 0.0:
        n = surf_local / (np.linalg.norm(surf_local) + 1e-9)
        surf_local = surf_local + n * touch_offset
    return center + rot @ surf_local


def solve_pin_ik_pose(
    pin_module: object,
    model: object,
    data: object,
    frame_id: int,
    q_init: np.ndarray,
    target_pose: object,
    max_iters: int,
    damping: float,
    step_size: float,
    pos_tol: float,
    ori_tol: float,
    pos_w: float,
    ori_w: float,
    fallback_q: np.ndarray | None,
) -> Tuple[np.ndarray, float, float, bool]:
    q = q_init.astype(np.float64, copy=True)

    def pose_err(vec_q: np.ndarray) -> np.ndarray:
        pin_module.forwardKinematics(model, data, vec_q)
        pin_module.updateFramePlacement(model, data, frame_id)
        cur = data.oMf[frame_id]
        dM = cur.actInv(target_pose)
        return pin_module.log6(dM).vector

    for _ in range(max_iters):
        err6 = pose_err(q)
        if not np.all(np.isfinite(err6)):
            q_fb = fallback_q if fallback_q is not None else q_init
            return q_fb.astype(np.float32), float("inf"), float("inf"), False
        ori_err = float(np.linalg.norm(err6[:3]))
        pos_err = float(np.linalg.norm(err6[3:]))
        if pos_err < pos_tol and ori_err < max(ori_tol, 0.35):
            return q.astype(np.float32), pos_err, ori_err, True

        # 使用 Pinocchio 解析雅可比，替换数值 Jacobian。
        pin_module.forwardKinematics(model, data, q)
        pin_module.updateFramePlacement(model, data, frame_id)
        cur = data.oMf[frame_id]
        dM = cur.actInv(target_pose)
        jac_local = pin_module.computeFrameJacobian(
            model,
            data,
            q,
            frame_id,
            pin_module.ReferenceFrame.LOCAL,
        )
        jlog = pin_module.Jlog6(dM)
        jac6 = -jlog @ jac_local

        w = np.array([ori_w, ori_w, ori_w, pos_w, pos_w, pos_w], dtype=np.float64)
        jw = jac6 * w[:, None]
        ew = err6 * w
        jj_t = jw @ jw.T
        try:
            dq = -jw.T @ np.linalg.solve(jj_t + damping * np.eye(6, dtype=np.float64), ew)
        except np.linalg.LinAlgError:
            q_fb = fallback_q if fallback_q is not None else q_init
            return q_fb.astype(np.float32), float("inf"), float("inf"), False
        if not np.all(np.isfinite(dq)):
            q_fb = fallback_q if fallback_q is not None else q_init
            return q_fb.astype(np.float32), float("inf"), float("inf"), False
        q = q + step_size * dq
        q = np.minimum(np.maximum(q, model.lowerPositionLimit), model.upperPositionLimit)

    # 回退：如果6D不收敛，执行位置IK，姿态作为软约束。
    target_t = target_pose.translation
    for _ in range(max(10, max_iters // 2)):
        pin_module.forwardKinematics(model, data, q)
        pin_module.updateFramePlacement(model, data, frame_id)
        cur_t = data.oMf[frame_id].translation
        pos_e = target_t - cur_t
        pos_n = float(np.linalg.norm(pos_e))
        if pos_n < pos_tol:
            err6 = pose_err(q)
            return q.astype(np.float32), pos_n, float(np.linalg.norm(err6[:3])), True

        jac6 = pin_module.computeFrameJacobian(
            model,
            data,
            q,
            frame_id,
            pin_module.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        # 兼容不同行顺序实现，选择对当前位置误差投影更有效的一组。
        cand_rows = [jac6[:3, :], jac6[3:, :]]
        best_jpos = cand_rows[0]
        best_gain = -1.0
        for jpos_c in cand_rows:
            gain = float(np.linalg.norm(jpos_c.T @ pos_e))
            if gain > best_gain:
                best_gain = gain
                best_jpos = jpos_c
        jpos = best_jpos

        jj_t = jpos @ jpos.T
        try:
            dq = jpos.T @ np.linalg.solve(jj_t + damping * np.eye(3, dtype=np.float64), pos_e)
        except np.linalg.LinAlgError:
            q_fb = fallback_q if fallback_q is not None else q_init
            return q_fb.astype(np.float32), float("inf"), float("inf"), False
        if not np.all(np.isfinite(dq)):
            q_fb = fallback_q if fallback_q is not None else q_init
            return q_fb.astype(np.float32), float("inf"), float("inf"), False
        q = q + step_size * dq
        q = np.minimum(np.maximum(q, model.lowerPositionLimit), model.upperPositionLimit)

    err6 = pose_err(q)
    pos_n = float(np.linalg.norm(target_t - data.oMf[frame_id].translation))
    return q.astype(np.float32), pos_n, float(np.linalg.norm(err6[:3])), False


def solve_pin_ik_position_fast(
    pin_module: object,
    model: object,
    data: object,
    frame_id: int,
    q_init: np.ndarray,
    target_world_xyz: np.ndarray,
    max_iters: int,
    damping: float,
    step_size: float,
    pos_tol: float,
) -> Tuple[np.ndarray, float, bool]:
    q = q_init.astype(np.float64, copy=True)
    for _ in range(max_iters):
        pin_module.forwardKinematics(model, data, q)
        pin_module.updateFramePlacement(model, data, frame_id)
        ee_pos = data.oMf[frame_id].translation
        err = target_world_xyz - ee_pos
        err_norm = float(np.linalg.norm(err))
        if err_norm < pos_tol:
            return q.astype(np.float32), err_norm, True

        jac6 = pin_module.computeFrameJacobian(
            model,
            data,
            q,
            frame_id,
            pin_module.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        # Pinocchio不同行版本在行顺序可能有差异，取两种候选中更稳定的一种。
        cand_rows = [jac6[:3, :], jac6[3:, :]]
        best_dq = None
        best_gain = -1.0
        for jpos in cand_rows:
            jj_t = jpos @ jpos.T
            dq = jpos.T @ np.linalg.solve(jj_t + damping * np.eye(3, dtype=np.float64), err)
            gain = float(np.linalg.norm(jpos @ dq))
            if gain > best_gain:
                best_gain = gain
                best_dq = dq
        if best_dq is None:
            break
        q = q + step_size * best_dq
        q = np.minimum(np.maximum(q, model.lowerPositionLimit), model.upperPositionLimit)

    pin_module.forwardKinematics(model, data, q)
    pin_module.updateFramePlacement(model, data, frame_id)
    final_err = float(np.linalg.norm(target_world_xyz - data.oMf[frame_id].translation))
    return q.astype(np.float32), final_err, False


def _has_unwanted_collision(model: mujoco.MjModel, data: mujoco.MjData, box_body_id: int) -> bool:
    # 过滤明显不合理碰撞：机器人与非box物体强碰撞，或机器人自碰撞。
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        if b1 <= 0 or b2 <= 0:
            continue
        is_box_contact = (b1 == box_body_id) or (b2 == box_body_id)
        if not is_box_contact:
            return True
    return False


def _contact_signal(model: mujoco.MjModel, data: mujoco.MjData, box_body_id: int) -> float:
    if model.nsensordata > 0:
        s = data.sensordata[: model.nsensordata]
        return float(np.max(np.abs(s))) if s.size > 0 else 0.0

    max_force = 0.0
    force6 = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        if b1 == box_body_id or b2 == box_body_id:
            mujoco.mj_contactForce(model, data, i, force6)
            max_force = max(max_force, float(np.linalg.norm(force6[:3])))
    return max_force


def _has_heavy_or_obstacle_collision(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    box_body_id: int,
    obstacle_geom_id: int,
    contact_force_reject: float,
) -> bool:
    if data.ncon == 0:
        return False

    force6 = np.zeros(6, dtype=np.float64)
    total_force = 0.0
    n_force = 0

    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        is_box_contact = (b1 == box_body_id) or (b2 == box_body_id)
        if is_box_contact:
            continue

        if obstacle_geom_id >= 0 and (g1 == obstacle_geom_id or g2 == obstacle_geom_id):
            return True

        mujoco.mj_contactForce(model, data, i, force6)
        total_force += float(np.linalg.norm(force6[:3]))
        n_force += 1

    if n_force == 0:
        return False
    return (total_force / n_force) > contact_force_reject


def collect_dataset(
    scene_xml_path: str,
    pin_mjcf_path: str,
    episodes: int,
    steps_per_episode: int,
    seed: int,
    box_low: np.ndarray,
    box_high: np.ndarray,
    box_vel_max: float,
    box_acc_std: float,
    static_box_prob: float,
    static_box_vel_max: float,
    target_z_offset: float,
    touch_offset: float,
    touch_dist_thresh: float,
    min_touch_ratio: float,
    target_filter_alpha: float,
    ik_max_iters: int,
    ik_damping: float,
    ik_step_size: float,
    ik_pos_tol: float,
    ik_ori_tol: float,
    ik_pos_w: float,
    ik_ori_w: float,
    fixed_gripper_cmd: float,
    obs_noise_std: float,
    max_episode_jerk: float,
    keep_dist_thresh: float,
    min_ik_success_rate: float,
    reject_collision: bool,
    contact_force_reject: float,
    ik_call_period: int,
    min_episode_samples: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    pin = _load_pinocchio()
    rng = np.random.default_rng(seed)

    model = mujoco.MjModel.from_xml_path(scene_xml_path)
    data = mujoco.MjData(model)
    joint_count = min(6, model.nu)
    ctrl_range = model.actuator_ctrlrange[:joint_count].copy()

    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    if box_body_id < 0:
        raise ValueError("scene 中未找到 body: box")
    if ee_site_id < 0:
        raise ValueError("scene 中未找到 site: gripperframe")

    box_joint_id = int(model.body_jntadr[box_body_id])
    if box_joint_id < 0 or model.jnt_type[box_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("box body 未挂载 freejoint，无法随机化世界坐标")

    box_qpos_adr = model.jnt_qposadr[box_joint_id]
    box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    if box_geom_id < 0:
        raise ValueError("scene 中未找到 geom: box")

    pin_model = pin.buildModelFromMJCF(pin_mjcf_path)
    pin_data = pin_model.createData()
    ee_frame_id = pin_model.getFrameId("gripperframe")

    dt = model.opt.timestep if model.opt.timestep > 0.0 else 0.005

    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    fallback_obs: List[np.ndarray] = []
    fallback_act: List[np.ndarray] = []
    emergency_obs: List[np.ndarray] = []
    emergency_act: List[np.ndarray] = []

    q_guess = np.zeros(joint_count, dtype=np.float32)
    last_success_q = q_guess.copy()
    ik_success = 0
    ik_total = 0
    kept_episodes = 0

    obstacle_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tilted_cylinder")

    for _ in range(episodes):
        mujoco.mj_resetData(model, data)

        # 机械臂随机初始化，提升状态多样性
        rand_q = rng.uniform(ctrl_range[:, 0], ctrl_range[:, 1]).astype(np.float64)
        data.qpos[:joint_count] = rand_q
        data.qvel[:joint_count] = rng.normal(0.0, 0.15, size=joint_count)

        if obstacle_geom_id >= 0:
            model.geom_pos[obstacle_geom_id] = rng.uniform(
                low=np.array([0.20, -0.18, 0.04]),
                high=np.array([0.32, 0.18, 0.10]),
            )

        box_xyz = rng.uniform(box_low, box_high).astype(np.float32)
        this_vel_max = static_box_vel_max if rng.random() < static_box_prob else box_vel_max
        box_vel = rng.uniform(-this_vel_max, this_vel_max, size=3).astype(np.float32)
        box_vel[2] = 0.0
        box_yaw = rng.uniform(-math.pi, math.pi)
        box_quat = np.array([math.cos(box_yaw * 0.5), 0.0, 0.0, math.sin(box_yaw * 0.5)], dtype=np.float64)
        data.qpos[box_qpos_adr:box_qpos_adr + 3] = box_xyz
        data.qpos[box_qpos_adr + 3:box_qpos_adr + 7] = box_quat
        box_dof_adr = int(model.jnt_dofadr[box_joint_id])
        data.qvel[box_dof_adr:box_dof_adr + 3] = box_vel
        data.qvel[box_dof_adr + 3:box_dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)

        ep_obs: List[np.ndarray] = []
        ep_act: List[np.ndarray] = []
        ep_q: List[np.ndarray] = []
        ep_ik_ok = 0
        ep_touch_ok = 0
        prev_ee = data.site_xpos[ee_site_id].copy()
        prev_box = data.xpos[box_body_id].copy()
        ep_invalid = False
        target_track: np.ndarray | None = None

        for t in range(steps_per_episode):
            # box 连续随机游走轨迹（平滑运动）
            box_acc = rng.normal(0.0, box_acc_std, size=3).astype(np.float32)
            box_acc[2] = 0.0
            box_vel = np.clip(box_vel + box_acc * dt, -this_vel_max, this_vel_max)
            box_xyz = box_xyz + box_vel * dt
            for k in range(3):
                if box_xyz[k] < box_low[k]:
                    box_xyz[k] = box_low[k]
                    box_vel[k] = abs(box_vel[k])
                elif box_xyz[k] > box_high[k]:
                    box_xyz[k] = box_high[k]
                    box_vel[k] = -abs(box_vel[k])

            data.qpos[box_qpos_adr:box_qpos_adr + 3] = box_xyz
            data.qpos[box_qpos_adr + 3:box_qpos_adr + 7] = box_quat
            data.qvel[box_dof_adr:box_dof_adr + 3] = box_vel
            data.qvel[box_dof_adr + 3:box_dof_adr + 6] = 0.0
            mujoco.mj_forward(model, data)

            cur_ee = data.site_xpos[ee_site_id].copy()
            cur_box = data.xpos[box_body_id].copy()
            ee_vel = ((cur_ee - prev_ee) / max(dt, 1e-6)).astype(np.float32)
            box_vel_obs = ((cur_box - prev_box) / max(dt, 1e-6)).astype(np.float32)
            prev_ee = cur_ee
            prev_box = cur_box

            obs = build_obs(
                data,
                joint_count,
                box_body_id,
                ee_site_id,
                box_vel=box_vel_obs,
                ee_vel=ee_vel,
                obs_noise_std=obs_noise_std,
                rng=rng,
            )

            target_pos = _box_touch_target_world(
                model=model,
                data=data,
                box_body_id=box_body_id,
                box_geom_id=box_geom_id,
                ee_world=cur_ee,
                z_offset=target_z_offset,
                touch_offset=touch_offset,
            )
            if target_track is None:
                target_track = target_pos.copy()
            else:
                a = float(np.clip(target_filter_alpha, 0.0, 1.0))
                target_track = (1.0 - a) * target_track + a * target_pos
            target_pos = target_track
            # 减少 IK 调用频率：每 N 步调用一次，其余步复用最近解。
            run_ik = (t % max(1, ik_call_period) == 0)
            if run_ik:
                q_ik, pos_err, ok = solve_pin_ik_position_fast(
                    pin_module=pin,
                    model=pin_model,
                    data=pin_data,
                    frame_id=ee_frame_id,
                    q_init=q_guess,
                    target_world_xyz=target_pos,
                    max_iters=ik_max_iters,
                    damping=ik_damping,
                    step_size=ik_step_size,
                    pos_tol=ik_pos_tol,
                )
            else:
                q_ik = q_guess.copy()
                pos_err = float(np.linalg.norm(target_pos - cur_ee))
                ok = pos_err < keep_dist_thresh

            touch_dist = float(np.linalg.norm(target_pos - cur_ee))
            if touch_dist <= touch_dist_thresh:
                ep_touch_ok += 1
            ik_total += 1
            ik_ok = ok or (pos_err < keep_dist_thresh)
            if ik_ok:
                ik_success += 1
                ep_ik_ok += 1
                last_success_q = q_ik.copy()
            q_guess = q_ik

            act = np.clip(q_ik, ctrl_range[:, 0], ctrl_range[:, 1]).astype(np.float32)

            act[5] = float(np.clip(fixed_gripper_cmd, ctrl_range[5, 0], ctrl_range[5, 1]))
            ep_obs.append(obs)
            ep_act.append(act)
            ep_q.append(act[:joint_count].copy())

            data.ctrl[:joint_count] = act
            mujoco.mj_step(model, data)

            if reject_collision and _has_heavy_or_obstacle_collision(
                model,
                data,
                box_body_id=box_body_id,
                obstacle_geom_id=obstacle_geom_id,
                contact_force_reject=contact_force_reject,
            ):
                ep_invalid = True
                break

            if not np.isfinite(pos_err):
                q_guess = last_success_q.copy()
                ep_invalid = True
                break

        # 记录应急样本池：即使 episode 被判 invalid，也尽量保留已有前缀数据。
        if len(ep_obs) >= max(1, min_episode_samples):
            emergency_obs.extend(ep_obs)
            emergency_act.extend(ep_act)

        if ep_invalid or len(ep_obs) < max(1, min_episode_samples):
            continue

        fallback_obs.extend(ep_obs)
        fallback_act.extend(ep_act)

        q_arr = np.asarray(ep_q, dtype=np.float32)
        jerk_rms = 0.0
        if q_arr.shape[0] >= 4:
            vel = np.diff(q_arr, axis=0) / max(dt, 1e-6)
            acc = np.diff(vel, axis=0) / max(dt, 1e-6)
            jerk = np.diff(acc, axis=0) / max(dt, 1e-6)
            jerk_rms = float(np.sqrt(np.mean(jerk**2)))

        final_dist = float(np.linalg.norm(data.xpos[box_body_id] - data.site_xpos[ee_site_id]))
        ep_ik_rate = ep_ik_ok / max(1, len(ep_obs))
        ep_touch_ratio = ep_touch_ok / max(1, len(ep_obs))
        if (
            jerk_rms > max_episode_jerk
            or final_dist > keep_dist_thresh
            or ep_ik_rate < min_ik_success_rate
            or ep_touch_ratio < min_touch_ratio
        ):
            continue

        obs_list.extend(ep_obs)
        act_list.extend(ep_act)
        kept_episodes += 1

    if not obs_list:
        if not fallback_obs:
            if emergency_obs:
                print("警告：严格筛选和回退筛选均为空，使用应急样本继续训练。")
                obs_list = emergency_obs
                act_list = emergency_act
            else:
                print("警告：未收集到有效样本，启动宽松兜底采样。")
                mujoco.mj_resetData(model, data)
                mujoco.mj_forward(model, data)
                prev_ee = data.site_xpos[ee_site_id].copy()
                prev_box = data.xpos[box_body_id].copy()
                for _ in range(max(20, steps_per_episode // 4)):
                    cur_ee = data.site_xpos[ee_site_id].copy()
                    cur_box = data.xpos[box_body_id].copy()
                    ee_vel = ((cur_ee - prev_ee) / max(dt, 1e-6)).astype(np.float32)
                    box_vel_obs = ((cur_box - prev_box) / max(dt, 1e-6)).astype(np.float32)
                    prev_ee = cur_ee
                    prev_box = cur_box

                    obs = build_obs(
                        data,
                        joint_count,
                        box_body_id,
                        ee_site_id,
                        box_vel=box_vel_obs,
                        ee_vel=ee_vel,
                        obs_noise_std=obs_noise_std,
                        rng=rng,
                    )
                    target_pos = _desired_target_pos(cur_box, target_z_offset)
                    q_ik, _, _ = solve_pin_ik_position_fast(
                        pin_module=pin,
                        model=pin_model,
                        data=pin_data,
                        frame_id=ee_frame_id,
                        q_init=q_guess,
                        target_world_xyz=target_pos,
                        max_iters=max(8, ik_max_iters // 3),
                        damping=ik_damping,
                        step_size=ik_step_size,
                        pos_tol=ik_pos_tol,
                    )
                    q_guess = q_ik
                    act = np.clip(q_ik, ctrl_range[:, 0], ctrl_range[:, 1]).astype(np.float32)
                    act[5] = float(np.clip(fixed_gripper_cmd, ctrl_range[5, 0], ctrl_range[5, 1]))
                    obs_list.append(obs)
                    act_list.append(act)
                    data.ctrl[:joint_count] = act
                    mujoco.mj_step(model, data)
        else:
            print("警告：严格轨迹筛选后样本为空，自动回退到未筛选有效样本。")
            obs_list = fallback_obs
            act_list = fallback_act
    obs_np = np.stack(obs_list, axis=0).astype(np.float32)
    act_np = np.stack(act_list, axis=0).astype(np.float32)
    success_rate = ik_success / max(1, ik_total)
    print(f"Pinocchio IK 成功率: {success_rate:.2%} ({ik_success}/{ik_total})")
    print(f"保留 episode: {kept_episodes}/{episodes}")
    return obs_np, act_np, joint_count


def _build_mlp(
    torch_module: object,
    in_dim: int,
    hidden: List[int],
    out_dim: int,
    dropout: float,
) -> object:
    nn = torch_module.nn
    layers: List[object] = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        layers.append(nn.Tanh())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def train_bc_policy(
    obs: np.ndarray,
    act: np.ndarray,
    hidden: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_ratio: float,
    early_stop_patience: int,
    early_stop_min_epochs: int,
    lr_patience: int,
    lr_factor: float,
    action_weights: np.ndarray,
    smooth_loss_coef: float,
    smooth_loss_adapt: bool,
    smooth_loss_min: float,
    smooth_loss_max: float,
    smooth_target_jump: float,
    smooth_adapt_gain: float,
    early_stop_min_delta: float,
    dropout: float,
    seed: int,
) -> Tuple[object, List[float], List[float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少 PyTorch，请先安装 torch。") from exc

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = obs.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    n_val = max(1, int(math.floor(n * val_ratio)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if train_idx.size == 0:
        raise ValueError("训练集为空，请减小 val_ratio。")

    obs_mean = obs[train_idx].mean(axis=0).astype(np.float32)
    obs_std = obs[train_idx].std(axis=0).astype(np.float32)
    obs_std = np.maximum(obs_std, 1e-6)

    act_mean = act[train_idx].mean(axis=0).astype(np.float32)
    act_std = act[train_idx].std(axis=0).astype(np.float32)
    act_std = np.maximum(act_std, 1e-6)

    obs_n = ((obs - obs_mean) / obs_std).astype(np.float32)
    act_n = ((act - act_mean) / act_std).astype(np.float32)

    x_train = torch.from_numpy(obs_n[train_idx]).to(device)
    y_train = torch.from_numpy(act_n[train_idx]).to(device)
    x_val = torch.from_numpy(obs_n[val_idx]).to(device)
    y_val = torch.from_numpy(act_n[val_idx]).to(device)

    model = _build_mlp(
        torch,
        in_dim=obs.shape[1],
        hidden=hidden,
        out_dim=act.shape[1],
        dropout=dropout,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
    )

    aw = torch.from_numpy(action_weights.astype(np.float32)).to(device)

    def weighted_mse(pred: object, target: object) -> object:
        return (((pred - target) ** 2) * aw).mean()

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    smooth_coef = float(smooth_loss_coef)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(x_train.shape[0])
        total_loss = 0.0

        for i in range(0, perm.shape[0], batch_size):
            b = perm[i:i + batch_size]
            xb = x_train[b]
            yb = y_train[b]

            pred = model(xb)
            loss = weighted_mse(pred, yb)
            if smooth_loss_coef > 0.0 and pred.shape[0] > 1:
                smooth = ((pred[1:] - pred[:-1]) ** 2).mean()
                loss = loss + smooth_coef * smooth

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss.detach().cpu().item()) * xb.shape[0]

        train_loss = total_loss / x_train.shape[0]
        train_loss_history.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = float(weighted_mse(val_pred, y_val).detach().cpu().item())
            if val_pred.shape[0] > 1:
                val_jump = float(((val_pred[1:] - val_pred[:-1]) ** 2).mean().detach().cpu().item())
            else:
                val_jump = 0.0
        val_loss_history.append(val_loss)
        scheduler.step(val_loss)

        if smooth_loss_adapt:
            err = val_jump - smooth_target_jump
            smooth_coef = float(np.clip(smooth_coef + smooth_adapt_gain * err, smooth_loss_min, smooth_loss_max))

        if val_loss < (best_val - max(0.0, early_stop_min_delta)):
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            cur_lr = float(optim.param_groups[0]["lr"])
            print(f"[Epoch {ep:04d}] train={train_loss:.6f} val={val_loss:.6f} jump={val_jump:.6f} smooth={smooth_coef:.4f} lr={cur_lr:.2e}")

        if ep >= max(1, early_stop_min_epochs) and bad_epochs >= early_stop_patience:
            print(f"Early stop at epoch={ep}, best_val={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_loss_history, val_loss_history, obs_mean, obs_std, act_mean, act_std


def plot_loss_curve(train_history: List[float], val_history: List[float], out_path: str, show_plot: bool) -> None:
    if not train_history:
        return
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少 matplotlib，请先安装 matplotlib。") from exc

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(train_history) + 1)
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(epochs, train_history, linewidth=2.0, label="train_loss")
    if val_history:
        plt.plot(np.arange(1, len(val_history) + 1), val_history, linewidth=2.0, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BC Training / Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"损失图已保存: {out_file}")

    if show_plot:
        plt.show()
    plt.close()


def export_to_npz(
    model: object,
    out_npz: str,
    tanh_out: bool,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    act_mean: np.ndarray,
    act_std: np.ndarray,
) -> None:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少 PyTorch，请先安装 torch。") from exc

    out_path = Path(out_npz)

    linear_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            linear_layers.append(m)

    if not linear_layers:
        raise ValueError("模型中未找到 Linear 层。")

    payload = {}
    for i, layer in enumerate(linear_layers):
        # torch: [out, in] -> stm2sim: [in, out]
        w = layer.weight.detach().cpu().numpy().astype(np.float32).T
        b = layer.bias.detach().cpu().numpy().astype(np.float32)
        payload[f"W{i}"] = w
        payload[f"b{i}"] = b

    action_dim = payload[f"b{len(linear_layers)-1}"].shape[0]
    payload["obs_mean"] = obs_mean.astype(np.float32)
    payload["obs_std"] = obs_std.astype(np.float32)
    payload["act_mean"] = act_mean.astype(np.float32)
    payload["act_std"] = act_std.astype(np.float32)
    # 兼容旧加载逻辑
    payload["act_scale"] = act_std.astype(np.float32)
    payload["act_bias"] = act_mean.astype(np.float32)
    payload["tanh_out"] = np.array(1 if tanh_out else 0, dtype=np.int32)

    np.savez(out_path, **payload)
    print(f"导出完成: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="基于 Pinocchio 逆解算的抓取离线策略训练并导出 NPZ")
    p.add_argument("--xml", type=str, default="scene_box.xml", help="MuJoCo 场景 XML（包含 box）")
    p.add_argument("--pin-mjcf", type=str, default="so101.xml", help="Pinocchio 用机器人 MJCF")
    p.add_argument("--output", type=str, default="offline_policy.npz", help="输出 NPZ 文件")
    p.add_argument("--episodes", type=int, default=80, help="采样 episode 数")
    p.add_argument("--steps", type=int, default=250, help="每个 episode 仿真步数")
    p.add_argument("--box-low", type=str, default="0.16,-0.16,0.03", help="box 世界坐标下界 x,y,z")
    p.add_argument("--box-high", type=str, default="0.32,0.16,0.03", help="box 世界坐标上界 x,y,z")
    p.add_argument("--box-vel-max", type=float, default=0.12, help="box 最大平移速度")
    p.add_argument("--static-box-prob", type=float, default=0.2, help="静止/低速box采样概率")
    p.add_argument("--static-box-vel-max", type=float, default=0.01, help="静止/低速box最大速度")
    p.add_argument("--box-acc-std", type=float, default=0.45, help="box 随机加速度标准差")
    p.add_argument("--target-z-offset", type=float, default=0.0, help="接近目标的 z 偏移")
    p.add_argument("--touch-offset", type=float, default=0.0, help="触碰目标点沿法向外偏移")
    p.add_argument("--touch-dist-thresh", type=float, default=0.05, help="判定触碰成功的距离阈值")
    p.add_argument("--min-touch-ratio", type=float, default=0.05, help="episode 最小持续触碰比例")
    p.add_argument("--target-filter-alpha", type=float, default=0.2, help="触碰目标点低通滤波系数")
    p.add_argument("--ik-iters", type=int, default=30, help="Pinocchio IK 最大迭代")
    p.add_argument("--ik-damping", type=float, default=1e-4, help="Pinocchio IK 阻尼")
    p.add_argument("--ik-step", type=float, default=0.4, help="Pinocchio IK 更新步长")
    p.add_argument("--ik-call-period", type=int, default=2, help="每 N 步调用一次 IK")
    p.add_argument("--ik-pos-tol", type=float, default=2e-2, help="Pinocchio IK 位置误差阈值")
    p.add_argument("--gripper-cmd", type=float, default=-0.05, help="固定夹爪控制值")
    p.add_argument("--obs-noise-std", type=float, default=0.003, help="观测高斯噪声标准差")
    p.add_argument("--max-episode-jerk", type=float, default=2500.0, help="轨迹平滑筛选阈值")
    p.add_argument("--keep-dist-thresh", type=float, default=0.18, help="保留 episode 的末端-目标最终距离阈值")
    p.add_argument("--min-ik-success-rate", type=float, default=0.02, help="保留 episode 的最小 IK 成功率")
    p.add_argument("--min-episode-samples", type=int, default=4, help="单个 episode 至少保留的样本数")
    p.add_argument("--reject-collision", action="store_true", default=False, help="启用碰撞样本剔除")
    p.add_argument("--allow-collision", action="store_true", help="允许碰撞样本（调试用）")
    p.add_argument("--contact-force-reject", type=float, default=80.0, help="平均接触力超过该值剔除样本")
    p.add_argument("--hidden", type=str, default="128,128", help="MLP 隐层，如 256,256")
    p.add_argument("--epochs", type=int, default=120, help="训练轮数")
    p.add_argument("--batch-size", type=int, default=512, help="批大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay")
    p.add_argument("--dropout", type=float, default=0.0, help="MLP dropout")
    p.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    p.add_argument("--early-stop-patience", type=int, default=200, help="早停 patience")
    p.add_argument("--early-stop-min-epochs", type=int, default=300, help="最早触发早停的 epoch")
    p.add_argument("--early-stop-min-delta", type=float, default=1e-6, help="早停判定最小改进量")
    p.add_argument("--lr-patience", type=int, default=30, help="学习率调度 patience")
    p.add_argument("--lr-factor", type=float, default=0.5, help="学习率衰减倍率")
    p.add_argument("--action-weights", type=str, default="1,1,1,1,1,0.5", help="各关节损失权重")
    p.add_argument("--smooth-loss-coef", type=float, default=0.02, help="动作差分平滑损失系数")
    p.add_argument("--smooth-loss-adapt", action="store_true", help="启用平滑损失自适应")
    p.add_argument("--smooth-loss-min", type=float, default=0.002, help="平滑损失最小系数")
    p.add_argument("--smooth-loss-max", type=float, default=0.08, help="平滑损失最大系数")
    p.add_argument("--smooth-target-jump", type=float, default=0.03, help="验证集目标动作突变程度")
    p.add_argument("--smooth-adapt-gain", type=float, default=0.2, help="平滑损失自适应增益")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--loss-plot", type=str, default="training_loss.png", help="损失曲线输出图片路径")
    p.add_argument("--show-plot", action="store_true", help="训练结束后弹窗显示损失曲线")
    p.add_argument("--no-tanh-out", action="store_true", help="导出时关闭输出 tanh")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    if not hidden:
        raise ValueError("--hidden 至少包含一个隐层维度。")

    box_low = np.array([float(x) for x in args.box_low.split(",")], dtype=np.float32)
    box_high = np.array([float(x) for x in args.box_high.split(",")], dtype=np.float32)
    if box_low.shape[0] != 3 or box_high.shape[0] != 3:
        raise ValueError("--box-low/--box-high 必须为 x,y,z 三个值")

    action_weights = np.array([float(x) for x in args.action_weights.split(",")], dtype=np.float32)
    if action_weights.shape[0] != 6:
        raise ValueError("--action-weights 需要 6 个值，对应 6 个关节")

    obs, act, joint_count = collect_dataset(
        scene_xml_path=args.xml,
        pin_mjcf_path=args.pin_mjcf,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        seed=args.seed,
        box_low=box_low,
        box_high=box_high,
        box_vel_max=args.box_vel_max,
        box_acc_std=args.box_acc_std,
        static_box_prob=float(np.clip(args.static_box_prob, 0.0, 1.0)),
        static_box_vel_max=max(0.0, args.static_box_vel_max),
        target_z_offset=args.target_z_offset,
        touch_offset=args.touch_offset,
        touch_dist_thresh=args.touch_dist_thresh,
        min_touch_ratio=float(np.clip(args.min_touch_ratio, 0.0, 1.0)),
        target_filter_alpha=float(np.clip(args.target_filter_alpha, 0.0, 1.0)),
        ik_max_iters=args.ik_iters,
        ik_damping=args.ik_damping,
        ik_step_size=args.ik_step,
        ik_pos_tol=args.ik_pos_tol,
        ik_ori_tol=0.0,
        ik_pos_w=1.0,
        ik_ori_w=0.0,
        fixed_gripper_cmd=args.gripper_cmd,
        obs_noise_std=args.obs_noise_std,
        max_episode_jerk=args.max_episode_jerk,
        keep_dist_thresh=args.keep_dist_thresh,
        min_ik_success_rate=args.min_ik_success_rate,
        reject_collision=(args.reject_collision and (not args.allow_collision)),
        contact_force_reject=max(0.0, args.contact_force_reject),
        ik_call_period=max(1, args.ik_call_period),
        min_episode_samples=max(1, args.min_episode_samples),
    )

    print(f"离线数据集: obs={obs.shape}, act={act.shape}, joints={joint_count}")

    model, train_history, val_history, obs_mean, obs_std, act_mean, act_std = train_bc_policy(
        obs=obs,
        act=act,
        hidden=hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_epochs=args.early_stop_min_epochs,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        action_weights=action_weights,
        smooth_loss_coef=args.smooth_loss_coef,
        smooth_loss_adapt=args.smooth_loss_adapt,
        smooth_loss_min=args.smooth_loss_min,
        smooth_loss_max=args.smooth_loss_max,
        smooth_target_jump=args.smooth_target_jump,
        smooth_adapt_gain=args.smooth_adapt_gain,
        early_stop_min_delta=args.early_stop_min_delta,
        dropout=args.dropout,
        seed=args.seed,
    )

    plot_loss_curve(
        train_history=train_history,
        val_history=val_history,
        out_path=args.loss_plot,
        show_plot=args.show_plot,
    )

    export_to_npz(
        model=model,
        out_npz=args.output,
        tanh_out=not args.no_tanh_out,
        obs_mean=obs_mean,
        obs_std=obs_std,
        act_mean=act_mean,
        act_std=act_std,
    )


if __name__ == "__main__":
    main()
