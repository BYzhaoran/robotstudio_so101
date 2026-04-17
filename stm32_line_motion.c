/*
 * stm32_line_motion.c
 *
 * 目的：
 *  - 在 STM32 侧输入当前位置与目标位置（笛卡尔坐标）
 *  - 按给定速度生成直线轨迹
 *  - 在固定控制周期内输出每一步的期望位置
 *
 * 使用方式：
 *  1) 调用 Motion_Init(control_period_s)
 *  2) 调用 Motion_SetCurrentPosition(x, y, z)
 *  3) 调用 Motion_SetTargetPosition(x, y, z, speed_mps)
 *  4) 在定时器中断或主循环中周期调用 Motion_Update()
 *  5) 在 Motion_OnSetpoint() 中把期望位置送给你的 IK/电机控制
 *
 * 说明：
 *  - 本文件只负责“直线插值轨迹规划”，不包含 IK 与电机驱动细节。
 *  - 单位建议统一为米（m）与秒（s）。
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    float x;
    float y;
    float z;
} Vec3;

typedef struct {
    Vec3 start_pos;          /* 直线段起点 */
    Vec3 target_pos;         /* 直线段终点 */
    Vec3 current_setpoint;   /* 当前插值点 */

    float distance;          /* 起点到终点距离 */
    float speed_mps;         /* 轨迹速度 (m/s) */
    float total_time_s;      /* 直线段总时长 */
    float elapsed_s;         /* 已运行时间 */

    float control_period_s;  /* 控制周期 */
    bool  active;            /* 是否在执行轨迹 */
} LineMotionPlanner;

static LineMotionPlanner g_planner;

/* 用户可在其它文件中重写此回调。 */
void Motion_OnSetpoint(float x, float y, float z, bool done);

/* ----------- 内部小工具函数 ----------- */
static float clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static float vec3_distance(Vec3 a, Vec3 b)
{
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float dz = b.z - a.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

static Vec3 vec3_lerp(Vec3 a, Vec3 b, float t)
{
    Vec3 out;
    out.x = a.x + (b.x - a.x) * t;
    out.y = a.y + (b.y - a.y) * t;
    out.z = a.z + (b.z - a.z) * t;
    return out;
}

/* ----------- 用户需要调用的接口 ----------- */
void Motion_Init(float control_period_s)
{
    if (control_period_s <= 0.0f) {
        control_period_s = 0.005f; /* 默认 5ms */
    }

    g_planner.start_pos.x = 0.0f;
    g_planner.start_pos.y = 0.0f;
    g_planner.start_pos.z = 0.0f;

    g_planner.target_pos = g_planner.start_pos;
    g_planner.current_setpoint = g_planner.start_pos;

    g_planner.distance = 0.0f;
    g_planner.speed_mps = 0.05f;
    g_planner.total_time_s = 0.0f;
    g_planner.elapsed_s = 0.0f;

    g_planner.control_period_s = control_period_s;
    g_planner.active = false;
}

void Motion_SetCurrentPosition(float x, float y, float z)
{
    /* 外部（编码器/状态估计）传入当前位置 */
    g_planner.current_setpoint.x = x;
    g_planner.current_setpoint.y = y;
    g_planner.current_setpoint.z = z;
}

void Motion_SetTargetPosition(float x, float y, float z, float speed_mps)
{
    /* 新任务开始时：起点 = 当前点，终点 = 输入目标 */
    g_planner.start_pos = g_planner.current_setpoint;

    g_planner.target_pos.x = x;
    g_planner.target_pos.y = y;
    g_planner.target_pos.z = z;

    if (speed_mps <= 1e-6f) {
        speed_mps = 0.01f; /* 防止除零，给一个最小速度 */
    }
    g_planner.speed_mps = speed_mps;

    g_planner.distance = vec3_distance(g_planner.start_pos, g_planner.target_pos);
    g_planner.elapsed_s = 0.0f;

    if (g_planner.distance <= 1e-6f) {
        /* 起点终点几乎一致，不需要运动 */
        g_planner.total_time_s = 0.0f;
        g_planner.active = false;
        g_planner.current_setpoint = g_planner.target_pos;
        Motion_OnSetpoint(g_planner.current_setpoint.x,
                          g_planner.current_setpoint.y,
                          g_planner.current_setpoint.z,
                          true);
        return;
    }

    g_planner.total_time_s = g_planner.distance / g_planner.speed_mps;
    g_planner.active = true;
}

void Motion_Stop(void)
{
    g_planner.active = false;
}

bool Motion_IsActive(void)
{
    return g_planner.active;
}

Vec3 Motion_GetSetpoint(void)
{
    return g_planner.current_setpoint;
}

void Motion_Update(void)
{
    bool done = false;

    if (g_planner.active) {
        g_planner.elapsed_s += g_planner.control_period_s;

        if (g_planner.total_time_s <= 1e-6f) {
            g_planner.current_setpoint = g_planner.target_pos;
            g_planner.active = false;
            done = true;
        } else {
            const float t = clampf(g_planner.elapsed_s / g_planner.total_time_s, 0.0f, 1.0f);
            g_planner.current_setpoint = vec3_lerp(g_planner.start_pos, g_planner.target_pos, t);

            if (t >= 1.0f) {
                g_planner.active = false;
                done = true;
            }
        }
    }

    /* 每个控制周期都把期望点输出给外部控制器 */
    Motion_OnSetpoint(g_planner.current_setpoint.x,
                      g_planner.current_setpoint.y,
                      g_planner.current_setpoint.z,
                      done);
}

/*
 * 用户回调：把轨迹点送到你的控制链路
 * 你可以在这里做：
 *  1) IK 计算 -> 关节角
 *  2) 关节 PID/FOC 控制
 *  3) 通过 CAN/UART 下发给下位驱动
 */
__attribute__((weak)) void Motion_OnSetpoint(float x, float y, float z, bool done)
{
    (void)x;
    (void)y;
    (void)z;
    (void)done;
    /* 默认空实现，用户在其它文件中重写该函数 */
}

/*
 * 示例（伪代码）：
 *
 * int main(void)
 * {
 *     HAL_Init();
 *     Motion_Init(0.005f); // 5ms 控制周期
 *
 *     // 假设当前末端位置来自传感器/状态估计
 *     Motion_SetCurrentPosition(0.18f, 0.00f, 0.12f);
 *
 *     // 输入目标位置 + 速度，开始直线运动
 *     Motion_SetTargetPosition(0.22f, 0.08f, 0.10f, 0.03f);
 *
 *     while (1) {
 *         // 每 5ms 调用一次（推荐用定时器中断）
 *         Motion_Update();
 *         HAL_Delay(5);
 *     }
 * }
 */

