"""阶段3: waypoints → (steer, throttle) 几何控制器.

输入: 当前帧 ego 系 waypoints [10,2] (x前 y左, 米) + 当前车速。
GeometricController: Pure Pursuit + 可选横向偏移修正 + PID 控速 + 曲率前瞻限速。

输出动作约定与 MetaDrive env.step 一致: [steering, throttle_brake],
steering 正=左转 (rad), throttle_brake ∈ [-1,1] 正=油门 负=刹车。
"""
from __future__ import annotations

import numpy as np

import config


# ============================================================
# 公共几何工具
# ============================================================
def path_from_waypoints(wp_ego: np.ndarray):
    """waypoints [K,2] → 折线 (原点开头), 返回 (pts[K+1,2], 累计弧长[K+1])."""
    pts = np.vstack([np.zeros(2), np.asarray(wp_ego, dtype=float)])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    return pts, arc


def point_at_arc(pts: np.ndarray, arc: np.ndarray, s: float):
    """折线上按弧长取点 (超出末端则沿末段方向外推)."""
    if s <= 0:
        return pts[0].copy()
    if s >= arc[-1]:
        d = pts[-1] - pts[-2]
        n = np.linalg.norm(d)
        d = d / n if n > 1e-6 else np.array([1.0, 0.0])
        return pts[-1] + d * (s - arc[-1])
    i = int(np.searchsorted(arc, s, side="right") - 1)
    i = min(i, len(pts) - 2)
    frac = (s - arc[i]) / max(arc[i + 1] - arc[i], 1e-9)
    return pts[i] + frac * (pts[i + 1] - pts[i])


def polyline_curvature(pts: np.ndarray):
    """相邻三点外接圆曲率, 返回每个内点的 |kappa| (1/m)."""
    if len(pts) < 3:
        return np.zeros(max(len(pts) - 2, 0))
    a = np.linalg.norm(pts[1:-1] - pts[:-2], axis=1)
    b = np.linalg.norm(pts[2:] - pts[1:-1], axis=1)
    c = np.linalg.norm(pts[2:] - pts[:-2], axis=1)
    cross = np.abs(
        (pts[1:-1, 0] - pts[:-2, 0]) * (pts[2:, 1] - pts[:-2, 1])
        - (pts[1:-1, 1] - pts[:-2, 1]) * (pts[2:, 0] - pts[:-2, 0])
    )
    denom = a * b * c
    kappa = np.where(denom > 1e-9, 2.0 * cross / np.maximum(denom, 1e-9), 0.0)
    return kappa


class PID:
    def __init__(self, kp, ki, kd, integral_max=3.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_max = integral_max
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.last_error = None

    def update(self, error, dt):
        self.integral = float(np.clip(self.integral + error * dt,
                                      -self.integral_max, self.integral_max))
        d = 0.0 if self.last_error is None else (error - self.last_error) / max(dt, 1e-6)
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * d


# ============================================================
# 方案 A: 几何控制器 Pure Pursuit + PID
# ============================================================
class GeometricController:
    def __init__(self, params: dict | None = None):
        p = dict(config.GEO_PARAMS)
        if params:
            p.update(params)
        self.p = p
        self.speed_pid = PID(p["kp_speed"], p["ki_speed"], p["kd_speed"], p["integral_max"])
        self.last_info = {}

    def reset(self):
        self.speed_pid.reset()
        self.last_info = {}

    def compute(self, wp_ego: np.ndarray, speed_kmh: float, dt: float):
        p = self.p
        v = max(float(speed_kmh) / 3.6, 0.0)
        pts, arc = path_from_waypoints(wp_ego)

        # ---- 横向: Pure Pursuit ----
        look = float(np.clip(p["look_base"] + p["look_scale"] * v, p["look_min"], p["look_max"]))
        target = point_at_arc(pts, arc, look)
        lx, ly = float(target[0]), float(target[1])
        l2 = max(lx * lx + ly * ly, 1e-6)
        # 自行车模型前轮转角 (rad, 正=左)
        delta_rad = float(np.arctan2(2.0 * p["wheelbase"] * ly, l2))

        # 横向偏移修正 (可选, 默认关闭):
        # waypoints 是 ego 系相对当前位姿的, 车辆偏离路径时折线已隐含偏移,
        # Pure Pursuit 本身就会拉回; 再加横向修正会重复计数导致过度转向。
        # 仅当 waypoints 来自固定世界路径投影时才建议开启 kp_steer>0。
        lat_err = 0.0
        if p["kp_steer"] > 0:
            d2 = (pts[1:, 0] ** 2 + pts[1:, 1] ** 2)
            j = int(np.argmin(d2)) + 1
            k = min(j + 1, len(pts) - 1)
            seg = pts[k] - pts[j]
            seg_n = np.linalg.norm(seg)
            if seg_n > 1e-6:
                nx, ny = -seg[1] / seg_n, seg[0] / seg_n  # 左法向
                lat_err = float((0.0 - pts[j, 0]) * nx + (0.0 - pts[j, 1]) * ny)
            # 符号约定: 正=左转。车在路径左侧(lat_err>0)需右转 → 修正项为负
            delta_rad += p["kp_steer"] * float(np.arctan2(lat_err, max(look, 1.0)))
        # 前轮角(rad) → MetaDrive 归一化动作 (实际前轮角 = 动作 × max_steering)
        steer = delta_rad / np.radians(p["max_steering_deg"])
        steer = float(np.clip(steer, -p["max_steer"], p["max_steer"]))

        # ---- 纵向: 曲率前瞻限速 + PID ----
        kappa = polyline_curvature(pts)
        n_preview = max(1, int(p["curve_preview_s"] / config.WP_DT))
        kmax = float(np.max(kappa[:n_preview])) if len(kappa) else 0.0
        v_curve = np.sqrt(p["lat_acc_max"] / kmax) if kmax > 1e-4 else np.inf
        v_target = min(config.TARGET_SPEED_KMH / 3.6, float(v_curve))
        throttle = float(np.clip(self.speed_pid.update(v_target - v, dt), -1.0, 1.0))

        self.last_info = dict(lookahead=look, lat_err=lat_err, kappa_max=kmax,
                              v_target_kmh=v_target * 3.6, steer_pp=delta_rad)
        return [steer, throttle]
