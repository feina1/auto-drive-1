"""waypoints × 相机图像 可视化 (无头离屏, 不弹任何窗口).

坐标与投影约定 (与 make_waypoints / MetaDrive 一致, 后续合并勿改):
- waypoints: ego 系 (x 前, y 左) 地面点; 车辆 origin 在底盘中心, 实测离地 ~0.5644 m,
  故地面点在 ego(origin 系) 的 z = GROUND_Z_EGO, 由 capture_at_pose 运行时实测覆盖;
- 相机: MetaDrive DEFAULT_SENSOR_OFFSET/HPR 挂在 origin,
  = ego (前 0.8, 上 1.5), 光轴俯视 5° → 相机离地 = 1.5 - GROUND_Z_EGO ≈ 2.0644 m;
- 针孔: camera_fov=60 为水平 FOV, fx=fy=(W/2)/tan(fov/2), cx=W/2, cy=H/2;
  ego→相机 (X右, Y上, Z前): Z=fwd·cosβ-up·sinβ, Y=fwd·sinβ+up·cosβ, X=-left;
  u=cx+fx·X/Z, v=cy-fy·Y/Z (get_image 已垂直翻转, 行 0 = 顶部)。

数值校验 (不靠目视 alone):
- 引擎相机矩阵独立交叉校验: cam.getRelativePoint(世界地面点) →
  u=cx+fx·X/Y, v=cy-fy·Z/Y (Panda3D 传感器节点 x右/y前/z上),
  与手算公式逐点比对, 阈值 2 px (实测 ≤ 0.7 px);
- 地平线检查: 地面点必须位于 v_horizon = cy - fy·tan(5°) 下方。

注: 曾用 DepthCamera 深度缓冲校验, 但 Mac offscreen 深度几何不可靠
(官方 CI 亦禁用 depth 测试), 已放弃; 锥桶道具渲染位置同样不可信, 勿再使用。

用法:
    python3 viz_waypoints.py --frames 1850,1965     # dataset_waypoints.npz 的帧索引
    python3 viz_waypoints.py --frames 10 --data PATH --out DIR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import config
import traj_utils as tu

# ---- 相机参数 (与 env 配置一一对应, 改配置必须同步改这里) ----
CAM_W, CAM_H = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
CAM_FOV_DEG = config.IMAGE_FOV_DEG
CAM_OFFSET_EGO = np.array([0.8, 0.0, 1.5])   # (前, 左, 上), 相对车辆 origin
CAM_PITCH_DEG = -5.0                          # 负 = 俯视
GROUND_Z_EGO = -0.5644                        # 地面 z (origin 系), 运行时实测覆盖
XCHECK_TOL_PX = 2.0


def cam_intrinsics():
    fx = (CAM_W / 2.0) / np.tan(np.radians(CAM_FOV_DEG) / 2.0)
    return fx, fx, CAM_W / 2.0, CAM_H / 2.0


def project_ego_to_pixel(pts_ego: np.ndarray):
    """ego 系地面点 [N,2] → (uv[N,2], Z[N]); Z<=0 在相机后方, uv 无意义."""
    fwd, left = pts_ego[:, 0], pts_ego[:, 1]
    up = np.full(len(fwd), GROUND_Z_EGO)
    d = np.stack([fwd, left, up], axis=1) - CAM_OFFSET_EGO
    fwd, left, up = d[:, 0], d[:, 1], d[:, 2]
    b = np.radians(-CAM_PITCH_DEG)
    cb, sb = np.cos(b), np.sin(b)
    Z = fwd * cb - up * sb
    Y = fwd * sb + up * cb
    X = -left
    fx, fy, cx, cy = cam_intrinsics()
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cx + fx * X / Z
        v = cy - fy * Y / Z
    return np.stack([u, v], axis=1), Z


def capture_at_pose(env, x: float, y: float, heading: float, speed_kmh: float) -> np.ndarray:
    """把车传送到指定位姿并拍一帧 RGB (离屏). 返回 BGR uint8 数组.

    关键: 无 main_camera 时 reset_sensors 不挂车, 必须手动 track 后再 renderFrame;
    同时运行时实测 origin 离地高度更新 GROUND_Z_EGO。
    """
    global GROUND_Z_EGO
    from metadrive.constants import DEFAULT_SENSOR_OFFSET, DEFAULT_SENSOR_HPR

    env.reset(seed=0)
    v = env.agent
    v.set_position([float(x), float(y)])
    v.set_heading_theta(float(heading))
    spd = float(speed_kmh) / 3.6
    v.set_velocity([spd * np.cos(heading), spd * np.sin(heading)])
    env.step([0.0, 0.0])  # 物理同步到设定位姿
    GROUND_Z_EGO = -float(v.origin.getZ(env.engine.render))
    cam = env.engine.get_sensor("rgb_camera")
    cam.track(v.origin, DEFAULT_SENSOR_OFFSET, DEFAULT_SENSOR_HPR)
    env.engine.graphicsEngine.renderFrame()
    return np.array(cam.get_image(env.agent, mode="bgr"), copy=True)


def engine_cross_check(env, wp_ego: np.ndarray) -> np.ndarray:
    """用引擎相机矩阵独立投影, 返回逐点与手算的像素距离 (px).

    世界地面点 z=0; 引擎传感器节点约定 x右/y前/z上 → u=cx+fx·X/Y, v=cy-fy·Z/Y。
    """
    from panda3d.core import Point3

    rc = env.engine.get_sensor("rgb_camera")
    fx, fy, cx, cy = cam_intrinsics()
    uv_hand, _ = project_ego_to_pixel(wp_ego)
    # 用传送后实际位姿 (与相机刚性一致), 而非数据集名义位姿
    v = env.agent
    h = float(v.heading_theta)
    bx, by = float(v.position[0]), float(v.position[1])
    c, s = np.cos(h), np.sin(h)
    out = np.zeros(len(wp_ego))
    for k in range(len(wp_ego)):
        wx = bx + wp_ego[k, 0] * c - wp_ego[k, 1] * s
        wy = by + wp_ego[k, 0] * s + wp_ego[k, 1] * c
        p = rc.cam.getRelativePoint(env.engine.render, Point3(wx, wy, 0.0))
        u = cx + fx * p.getX() / p.getY()
        vv = cy - fy * p.getZ() / p.getY()
        out[k] = float(np.hypot(u - uv_hand[k, 0], vv - uv_hand[k, 1]))
    return out


def overlay(rgb: np.ndarray, wp_ego: np.ndarray, t: float, speed_kmh: float) -> np.ndarray:
    """在相机图上绘制 waypoints 折线+圆点+时间标签+地平线, 返回可写副本."""
    import cv2

    img = np.array(rgb, copy=True)
    uv, Z = project_ego_to_pixel(wp_ego)
    fx, fy, cx, cy = cam_intrinsics()
    v_h = cy - fy * np.tan(np.radians(-CAM_PITCH_DEG))
    cv2.line(img, (0, int(v_h)), (CAM_W, int(v_h)), (255, 255, 0), 1)
    pts = uv.astype(int)
    for k in range(len(pts) - 1):
        if Z[k] > 0 and Z[k + 1] > 0:
            cv2.line(img, tuple(pts[k]), tuple(pts[k + 1]), (0, 200, 200), 2)
    for k in range(len(pts)):
        u, vv = pts[k]
        color = (0, 255, 0) if 0 <= u < CAM_W and 0 <= vv < CAM_H else (0, 0, 255)
        cv2.circle(img, (u, vv), 6, color, -1)
        cv2.circle(img, (u, vv), 6, (255, 255, 255), 1)
        cv2.putText(img, f"{(k + 1) * 0.5:.1f}s", (u + 8, vv - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"t={t:.1f}s  v={speed_kmh:.0f}km/h  wp=ego(x fwd, y left)",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True, help="逗号分隔的帧索引")
    parser.add_argument("--data", type=str, default=str(config.OUT_DIR / "dataset_waypoints.npz"))
    parser.add_argument("--out", type=str, default=str(config.OUT_DIR / "viz"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import cv2

    z = np.load(args.data)
    idxs = [int(s) for s in args.frames.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = tu.build_headless_env(with_camera=True)
    all_ok = True
    try:
        for i in idxs:
            rgb = capture_at_pose(env, float(z["x"][i]), float(z["y"][i]),
                                  float(z["heading"][i]), float(z["speed_kmh"][i]))
            wp = z["wp_ego"][i].astype(float)
            errs = engine_cross_check(env, wp)
            uv, _ = project_ego_to_pixel(wp)
            fx, fy, cx, cy = cam_intrinsics()
            v_h = cy - fy * np.tan(np.radians(-CAM_PITCH_DEG))
            below = bool(np.all(uv[wp[:, 0] > 0, 1] > v_h))
            ok = bool(np.all(errs < XCHECK_TOL_PX)) and below
            all_ok &= ok
            path = out_dir / f"viz_frame{i:05d}.png"
            cv2.imwrite(str(path), overlay(rgb, wp, float(z["t"][i]), float(z["speed_kmh"][i])))
            print(f"[viz] frame {i} t={z['t'][i]:.1f}s: 交叉校验 max={errs.max():.2f}px "
                  f"(tol {XCHECK_TOL_PX}), 地平线以下={below} → {'PASS' if ok else 'FAIL'} → {path}",
                  flush=True)
    finally:
        env.close()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
