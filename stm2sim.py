from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List

import mujoco
import mujoco.viewer


# ======================
# STM32 arm_shaper.c 移植
# ======================

ARM_POS_EPS = 1e-4
ARM_VEL_EPS = 1e-3
ARM_ACC_EPS = 1e-2


def clampf(x: float, xmin: float, xmax: float) -> float:
    if x > xmax:
        return xmax
    if x < xmin:
        return xmin
    return x


def signf_nonzero(x: float) -> float:
    return 1.0 if x >= 0.0 else -1.0


@dataclass
class ArmCmdProfile:
    pos: float = 0.0
    vel: float = 0.0
    acc: float = 0.0
    vel_cmd: float = 0.0
    vel_rate: float = 0.0
    torque_cmd: float = 0.0
    torque_rate: float = 0.0
    inited: int = 0


@dataclass(frozen=True)
class ArmScurveLimit:
    v_max: float
    a_max: float
    j_max: float


@dataclass(frozen=True)
class ArmSlewJerkLimit:
    slew: float
    jerk: float


@dataclass
class ArmCmdOutput:
    pos_des: float = 0.0
    vel_des: float = 0.0
    torque_des: float = 0.0


def pos_scurve_step(profile: ArmCmdProfile,
                    target_pos: float,
                    dt: float,
                    limit: ArmScurveLimit) -> None:
    pos_err = target_pos - profile.pos
    vel_brake = math.sqrt(2.0 * limit.a_max * abs(pos_err))
    vel_ref = signf_nonzero(pos_err) * (vel_brake if vel_brake < limit.v_max else limit.v_max)
    acc_ref = clampf((vel_ref - profile.vel) / dt, -limit.a_max, limit.a_max)
    dacc = clampf(acc_ref - profile.acc, -limit.j_max * dt, limit.j_max * dt)

    profile.acc += dacc
    profile.vel += profile.acc * dt
    profile.vel = clampf(profile.vel, -limit.v_max, limit.v_max)
    profile.pos += profile.vel * dt

    if (abs(target_pos - profile.pos) < ARM_POS_EPS
            and abs(profile.vel) < ARM_VEL_EPS
            and abs(profile.acc) < ARM_ACC_EPS):
        profile.pos = target_pos
        profile.vel = 0.0
        profile.acc = 0.0


def slew_jerk_step(target: float,
                   dt: float,
                   limit: ArmSlewJerkLimit,
                   state: List[float],
                   rate_state: List[float]) -> float:
    desired_rate = clampf((target - state[0]) / dt, -limit.slew, limit.slew)
    dr = clampf(desired_rate - rate_state[0], -limit.jerk * dt, limit.jerk * dt)
    rate_state[0] += dr
    state[0] += rate_state[0] * dt
    return state[0]


def arm_shaper_profile_init(profile: ArmCmdProfile, init_pos: float) -> None:
    profile.inited = 1
    profile.pos = init_pos
    profile.vel = 0.0
    profile.acc = 0.0
    profile.vel_cmd = 0.0
    profile.vel_rate = 0.0
    profile.torque_cmd = 0.0
    profile.torque_rate = 0.0


def arm_shaper_reset(profile: ArmCmdProfile) -> None:
    profile.inited = 0
    profile.vel = 0.0
    profile.acc = 0.0
    profile.vel_cmd = 0.0
    profile.vel_rate = 0.0
    profile.torque_cmd = 0.0
    profile.torque_rate = 0.0


def arm_shaper_sync_to_motor_pos(profile: ArmCmdProfile, motor_pos: float) -> None:
    profile.inited = 1
    profile.pos = motor_pos
    profile.vel = 0.0
    profile.acc = 0.0
    profile.vel_cmd = 0.0
    profile.vel_rate = 0.0
    profile.torque_cmd = 0.0
    profile.torque_rate = 0.0


def arm_shaper_step(profile: ArmCmdProfile,
                    target_pos: float,
                    torque_target: float,
                    dt: float,
                    pos_limit: ArmScurveLimit,
                    vel_limit: ArmSlewJerkLimit,
                    torque_limit: ArmSlewJerkLimit,
                    out: ArmCmdOutput) -> None:
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    if not profile.inited:
        arm_shaper_profile_init(profile, target_pos)

    pos_scurve_step(profile, target_pos, dt, pos_limit)

    vel_state = [profile.vel_cmd]
    vel_rate_state = [profile.vel_rate]
    profile.vel_cmd = slew_jerk_step(profile.vel, dt, vel_limit, vel_state, vel_rate_state)
    profile.vel_rate = vel_rate_state[0]

    tor_state = [profile.torque_cmd]
    tor_rate_state = [profile.torque_rate]
    profile.torque_cmd = slew_jerk_step(torque_target, dt, torque_limit, tor_state, tor_rate_state)
    profile.torque_rate = tor_rate_state[0]

    out.pos_des = profile.pos
    out.vel_des = profile.vel_cmd
    out.torque_des = profile.torque_cmd


# ======================
# MuJoCo 驱动示例（6关节）
# ======================

model = mujoco.MjModel.from_xml_path("scene_box.xml")
data = mujoco.MjData(model)

joint_count = min(6, model.nu)
profiles = [ArmCmdProfile() for _ in range(joint_count)]
outputs = [ArmCmdOutput() for _ in range(joint_count)]

# 下面参数可按你的STM32工程配置替换
pos_limit = ArmScurveLimit(v_max=1.5, a_max=3.0, j_max=800.0)
vel_limit = ArmSlewJerkLimit(slew=6.0, jerk=100.0)
tor_limit = ArmSlewJerkLimit(slew=10.0, jerk=300.0)

# 用于演示“大误差目标变化”
target_a = [0.0, -0.8, 1.2, 0.6, -0.4, 0.3]
target_b = [1.4, 0.7, 20.0, -0.5, 0.9, 0.3]

sim_time = 0.0
switch_period = 2.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        dt = model.opt.timestep
        if dt <= 0.0:
            dt = 0.001

        sim_time += dt
        use_b = int(sim_time / switch_period) % 2 == 1
        target = target_b if use_b else target_a

        for i in range(joint_count):
            arm_shaper_step(
                profile=profiles[i],
                target_pos=target[i],
                torque_target=0.0,
                dt=dt,
                pos_limit=pos_limit,
                vel_limit=vel_limit,
                torque_limit=tor_limit,
                out=outputs[i],
            )
            data.ctrl[i] = outputs[i].pos_des

        mujoco.mj_step(model, data)
        viewer.sync()