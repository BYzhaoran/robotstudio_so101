from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


@dataclass
class OfflinePolicy:
    """简单 MLP 离线策略推理器（权重来自 .npz）。"""

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    action_bias: np.ndarray
    action_scale: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    use_tanh_output: bool

    @staticmethod
    def from_npz(policy_path: str) -> "OfflinePolicy":
        p = Path(policy_path)
        if not p.exists():
            raise FileNotFoundError(
                f"未找到离线策略文件: {policy_path}。请先导出策略为 .npz。"
            )

        blob = np.load(p, allow_pickle=False)
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []

        layer_idx = 0
        while f"W{layer_idx}" in blob and f"b{layer_idx}" in blob:
            weights.append(np.array(blob[f"W{layer_idx}"], dtype=np.float32))
            biases.append(np.array(blob[f"b{layer_idx}"], dtype=np.float32))
            layer_idx += 1

        if not weights:
            raise ValueError("策略文件缺少网络参数，至少需要 W0/b0。")

        action_dim = biases[-1].shape[0]
        if "act_mean" in blob and "act_std" in blob:
            action_bias = np.array(blob["act_mean"], dtype=np.float32)
            action_scale = np.array(blob["act_std"], dtype=np.float32)
        else:
            action_bias = np.array(blob["act_bias"], dtype=np.float32) if "act_bias" in blob else np.zeros(action_dim, dtype=np.float32)
            action_scale = np.array(blob["act_scale"], dtype=np.float32) if "act_scale" in blob else np.ones(action_dim, dtype=np.float32)

        obs_dim = weights[0].shape[0]
        obs_mean = np.array(blob["obs_mean"], dtype=np.float32) if "obs_mean" in blob else np.zeros(obs_dim, dtype=np.float32)
        obs_std = np.array(blob["obs_std"], dtype=np.float32) if "obs_std" in blob else np.ones(obs_dim, dtype=np.float32)
        obs_std = np.maximum(obs_std, 1e-6)
        use_tanh_output = bool(blob["tanh_out"]) if "tanh_out" in blob else True

        return OfflinePolicy(
            weights=weights,
            biases=biases,
            action_bias=action_bias,
            action_scale=action_scale,
            obs_mean=obs_mean,
            obs_std=obs_std,
            use_tanh_output=use_tanh_output,
        )

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32, copy=False)
        exp_dim = int(self.obs_mean.shape[0])
        if obs.shape[0] > exp_dim:
            obs = obs[:exp_dim]
        elif obs.shape[0] < exp_dim:
            obs = np.concatenate([obs, np.zeros(exp_dim - obs.shape[0], dtype=np.float32)])

        x = (obs - self.obs_mean) / self.obs_std
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.tanh(x)
        if self.use_tanh_output:
            x = np.tanh(x)
        return x * self.action_scale + self.action_bias


def build_obs(
    data: mujoco.MjData,
    joint_count: int,
    target: np.ndarray,
    ee_site_id: int,
    box_vel: np.ndarray,
    ee_vel: np.ndarray,
    obs_noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    qpos = data.qpos[:joint_count]
    qvel = data.qvel[:joint_count]
    box_pos = target
    ee_pos = data.site_xpos[ee_site_id].copy()
    rel = box_pos - ee_pos
    box_vel = np.asarray(box_vel, dtype=np.float32)
    ee_vel = np.asarray(ee_vel, dtype=np.float32)
    rel_vel = box_vel - ee_vel
    obs = np.concatenate([qpos, qvel, box_pos, ee_pos, rel, box_vel, ee_vel, rel_vel], dtype=np.float32)
    if obs_noise_std > 0.0:
        obs = obs + rng.normal(0.0, obs_noise_std, size=obs.shape).astype(np.float32)
    return obs


def box_touch_target_world(
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
        margin = half - np.abs(rel_local)
        axis = int(np.argmin(margin))
        sgn = 1.0 if rel_local[axis] >= 0.0 else -1.0
        surf_local[axis] = sgn * half[axis]

    if touch_offset != 0.0:
        n = surf_local / (np.linalg.norm(surf_local) + 1e-9)
        surf_local = surf_local + n * touch_offset
    return center + rot @ surf_local


def detect_bad_collision(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    box_body_id: int,
    obstacle_geom_id: int,
    force_thresh: float,
) -> tuple[bool, float]:
    if data.ncon == 0:
        return False, 0.0

    f6 = np.zeros(6, dtype=np.float64)
    total = 0.0
    n = 0

    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])

        # 忽略与 box 的接触（任务需要）
        if b1 == box_body_id or b2 == box_body_id:
            continue

        # 机器人与障碍（tilted_cylinder）碰撞直接判危险
        if obstacle_geom_id >= 0 and (g1 == obstacle_geom_id or g2 == obstacle_geom_id):
            return True, force_thresh

        # 机器人自碰撞（两个都属于机器人 body）
        if b1 > 0 and b2 > 0:
            return True, force_thresh

        # 机器人与环境的强接触
        if (b1 > 0) ^ (b2 > 0):
            mujoco.mj_contactForce(model, data, i, f6)
            total += float(np.linalg.norm(f6[:3]))
            n += 1

    if n == 0:
        return False, 0.0
    mean_force = total / n
    return mean_force > force_thresh, mean_force


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用离线强化学习策略驱动 SO101")
    parser.add_argument("--xml", type=str, default="scene_box.xml", help="MuJoCo 场景 XML")
    parser.add_argument("--policy", type=str, default="offline_policy.npz", help="离线策略文件 (.npz)")
    parser.add_argument("--smooth", type=float, default=0.2, help="基础平滑系数，范围 [0,1]")
    parser.add_argument("--smooth-adapt-gain", type=float, default=0.8, help="按误差自适应调节平滑强度")
    parser.add_argument("--smooth-min", type=float, default=0.02, help="最小平滑系数")
    parser.add_argument("--smooth-max", type=float, default=0.95, help="最大平滑系数")
    parser.add_argument("--obs-noise-std", type=float, default=0.0, help="验证时观测噪声标准差")
    parser.add_argument("--qvel-limit", type=float, default=6.0, help="关节速度安全阈值")
    parser.add_argument("--stall-seconds", type=float, default=2.0, help="距离长期不下降触发恢复")
    parser.add_argument("--stall-eps", type=float, default=1e-3, help="停滞判断阈值")
    parser.add_argument("--target-z-offset", type=float, default=0.0, help="触碰目标点 z 偏移")
    parser.add_argument("--touch-offset", type=float, default=0.0, help="触碰目标点法向偏移")
    parser.add_argument("--touch-dist-thresh", type=float, default=0.02, help="持续触碰距离阈值")
    parser.add_argument("--target-filter-alpha", type=float, default=0.2, help="触碰目标点低通滤波系数")
    parser.add_argument("--gripper-cmd", type=float, default=-0.05, help="固定夹爪控制值")
    parser.add_argument("--collision-force-thresh", type=float, default=8.0, help="平均接触力超过该值触发恢复")
    parser.add_argument("--recovery-steps", type=int, default=120, help="碰撞后回退控制步数")
    parser.add_argument("--log-interval", type=float, default=0.5, help="日志输出间隔（秒）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smooth = float(np.clip(args.smooth, 0.0, 1.0))
    rng = np.random.default_rng(123)

    policy = OfflinePolicy.from_npz(args.policy)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    joint_count = min(6, model.nu)
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    obstacle_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tilted_cylinder")
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    if box_body_id < 0:
        raise ValueError("scene 中未找到 body: box")
    if box_geom_id < 0:
        raise ValueError("scene 中未找到 geom: box")
    if ee_site_id < 0:
        raise ValueError("scene 中未找到 site: gripperframe")

    ctrl_range = model.actuator_ctrlrange[:joint_count].copy()
    last_ctrl = np.zeros(joint_count, dtype=np.float32)
    sim_time = 0.0
    last_log_t = -1e9
    last_improve_t = 0.0
    best_recent_dist = float("inf")
    target_track: np.ndarray | None = None
    recovery_left = 0

    mujoco.mj_forward(model, data)
    prev_box = data.xpos[box_body_id].copy()
    prev_ee = data.site_xpos[ee_site_id].copy()
    home_qpos = data.qpos[:joint_count].copy().astype(np.float32)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            dt = model.opt.timestep if model.opt.timestep > 0.0 else 0.001
            sim_time += dt

            box_center = data.xpos[box_body_id].copy()
            ee_pos = data.site_xpos[ee_site_id].copy()
            raw_target = box_touch_target_world(
                model=model,
                data=data,
                box_body_id=box_body_id,
                box_geom_id=box_geom_id,
                ee_world=ee_pos,
                z_offset=args.target_z_offset,
                touch_offset=args.touch_offset,
            )
            if target_track is None:
                target_track = raw_target.copy()
            else:
                a = float(np.clip(args.target_filter_alpha, 0.0, 1.0))
                target_track = (1.0 - a) * target_track + a * raw_target
            target = target_track
            box_vel = (box_center - prev_box) / max(dt, 1e-6)
            ee_vel = (ee_pos - prev_ee) / max(dt, 1e-6)
            prev_box = box_center.copy()
            prev_ee = ee_pos.copy()

            obs = build_obs(
                data,
                joint_count,
                target,
                ee_site_id,
                box_vel=box_vel.astype(np.float32),
                ee_vel=ee_vel.astype(np.float32),
                obs_noise_std=args.obs_noise_std,
                rng=rng,
            )
            action = policy.act(obs)
            if action.shape[0] != joint_count:
                raise ValueError(
                    f"策略输出维度={action.shape[0]}，但当前关节数={joint_count}。"
                )

            dist = float(np.linalg.norm(target - ee_pos))
            smooth_eff = smooth - args.smooth_adapt_gain * min(dist, 0.4)
            smooth_eff = float(np.clip(smooth_eff, args.smooth_min, args.smooth_max))

            ctrl = (1.0 - smooth_eff) * action + smooth_eff * last_ctrl
            ctrl = np.clip(ctrl, ctrl_range[:, 0], ctrl_range[:, 1])
            ctrl[5] = float(np.clip(args.gripper_cmd, ctrl_range[5, 0], ctrl_range[5, 1]))

            bad_collision, mean_force = detect_bad_collision(
                model,
                data,
                box_body_id=box_body_id,
                obstacle_geom_id=obstacle_geom_id,
                force_thresh=args.collision_force_thresh,
            )
            if bad_collision:
                recovery_left = max(recovery_left, args.recovery_steps)

            if recovery_left > 0:
                # 回退到安全初始姿态，避免卡在自碰撞配置。
                ctrl = home_qpos.copy()
                ctrl[5] = ctrl_range[5, 0]
                recovery_left -= 1

            if float(np.max(np.abs(data.qvel[:joint_count]))) > args.qvel_limit:
                # 超速保护：切回位置保持
                ctrl = data.qpos[:joint_count].astype(np.float32)

            if dist + args.stall_eps < best_recent_dist:
                best_recent_dist = dist
                last_improve_t = sim_time
            elif sim_time - last_improve_t > args.stall_seconds and dist > args.touch_dist_thresh:
                # 恢复策略：短暂回到当前姿态并张开夹爪
                ctrl = data.qpos[:joint_count].astype(np.float32)
                ctrl[5] = ctrl_range[5, 0]
                last_improve_t = sim_time
                best_recent_dist = dist

            data.ctrl[:joint_count] = ctrl
            last_ctrl = ctrl

            if sim_time - last_log_t >= args.log_interval:
                print(
                    f"[sim {sim_time:7.3f}s] ee-box dist={dist:.4f} m, "
                    f"smooth={smooth_eff:.3f}, rec={recovery_left}, cF={mean_force:.2f}"
                )
                last_log_t = sim_time

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()