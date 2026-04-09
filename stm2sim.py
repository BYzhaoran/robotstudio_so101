from __future__ import annotations

from typing import Sequence

import mujoco
import mujoco.viewer


def apply_control(data: mujoco.MjData, targets: Sequence[float], joint_count: int) -> None:
    """最小控制框架：把目标写入控制输入。"""
    for i in range(joint_count):
        data.ctrl[i] = targets[i]


def main() -> None:
    # 模型调用
    model = mujoco.MjModel.from_xml_path("scene_box.xml")
    data = mujoco.MjData(model)

    # 控制框架
    joint_count = min(6, model.nu)
    targets = [0.0] * joint_count

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            with viewer.lock():
                apply_control(data, targets, joint_count)
                mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
