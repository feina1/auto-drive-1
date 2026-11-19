"""MetaDrive 开窗可视化 viewer: 10Hz tick, 每 tick 图→模型→waypoints→控制器→动作.

链路完全复用 waypoint_pipeline 里已有的组件, 不改动任何现有代码:
  - 模型加载与推理: training/closed_loop.py::Predictor
  - 控制器: controllers.py::GeometricController
  - RGB 抓取 / 参考线 / 横向误差: traj_utils.py
  - Waypoints 投影与画面叠加: viz_waypoints.py::overlay
  - 专家地图 / 圈数计数: simple_track.py (仓库根)

可视化面板:
  - 左侧 cv2 窗口: 前相机画面 + waypoints 折线/圆点/时间标签 (同 viz_waypoints.py)
  - 右侧 Panda3D 窗口: MetaDrive 3D 追车视角 (--panel-only 可关闭)

用法:
    python viewer.py --ckpt output/training/dinov3_best.pt --speed 56 --max-seconds 30
    python viewer.py --ckpt output/training/dinov3_best.pt --speed 40 --realtime
    python viewer.py --ckpt output/training/dinov3_best.pt --panel-only

参数:
    --speed         目标巡航速度 km/h (默认 56)
    --max-seconds   仿真时长上限, 0 = 不限 (默认 30s 冒烟)
    --realtime      按 10Hz 墙钟对齐 (默认尽可能快跑, 窗口刷新会跟不上)
    --device        auto/cpu/cuda/mps (默认 cuda)
    --seed          随机种子
    --panel-only    不弹 MetaDrive 3D 窗口, 只显示 waypoints 面板 (离屏渲染)

关闭窗口或 Ctrl+C 即可中断.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# 保证能 import 到 waypoint_pipeline 顶层与 training 子目录 (与 training/common.py 同套思路)
_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE / "training"), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config  # noqa: E402
import traj_utils as tu  # noqa: E402
import viz_waypoints as vw  # noqa: E402
from controllers import GeometricController  # noqa: E402
from closed_loop import Predictor  # noqa: E402
from common import get_device  # noqa: E402


def build_render_env(with_camera: bool, spawn_speed_kmh: float, use_render: bool = True):
    """与 traj_utils.build_headless_env 同参数, 支持 use_render 开关.

    注意: 窗口模式下 image_on_cuda / multi_thread_render 会干扰 Panda3D 主循环,
    这里强制关掉, 只保留 render_pipeline=False 走默认管线.
    """
    from simple_track import LANE_NUM, LANE_WIDTH, SimpleTrackEnv

    cfg = dict(
        use_render=use_render,
        manual_control=False,
        traffic_density=0.0,
        num_scenarios=1,
        start_seed=0,
        map_region_size=4096,
        show_interface=use_render,
        show_logo=False,
        show_fps=use_render,
        image_on_cuda=False,
        multi_thread_render=False,
        render_pipeline=False,
        random_lane_width=False,
        random_lane_num=False,
        out_of_route_done=False,
        on_continuous_line_done=False,
        out_of_road_done=False,
        crash_vehicle_done=False,
        crash_object_done=False,
        crash_human_done=False,
        map_config=dict(lane_num=LANE_NUM, lane_width=LANE_WIDTH),
        vehicle_config=dict(
            show_navi_mark=False,
            show_line_to_navi_mark=False,
            show_lidar=False,
            spawn_velocity=[spawn_speed_kmh / 3.6, 0.0],
            spawn_velocity_car_frame=True,
        ),
    )
    if with_camera:
        from metadrive.component.sensors.rgb_camera import RGBCamera

        cfg.update(
            image_observation=True,
            camera_fov=config.IMAGE_FOV_DEG,
            sensors=dict(rgb_camera=(RGBCamera, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)),
        )
        cfg["vehicle_config"]["image_source"] = "rgb_camera"
    return SimpleTrackEnv(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--speed", type=float, default=56.0, help="目标巡航速度 km/h")
    parser.add_argument("--max-seconds", type=float, default=30.0,
                        help="仿真时长上限, 0 = 不限")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--realtime", action="store_true",
                        help="按 10Hz 墙钟对齐, 每 tick 若算得快就 sleep 补齐")
    parser.add_argument("--panel-only", action="store_true",
                        help="不弹 MetaDrive 3D 窗口, 只显示 waypoints 面板 (离屏渲染)")
    args = parser.parse_args()

    from simple_track import LapCounter
    from PIL import Image
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    device = get_device(args.device)
    predictor = Predictor(args.ckpt, device)
    controller = GeometricController()

    use_render = not args.panel_only
    env = build_render_env(with_camera=True, spawn_speed_kmh=args.speed, use_render=use_render)
    tracker = tu.LateralErrorTracker(*tu.load_reference_path())
    aborted = None
    panel_fig = None
    panel_img = None

    try:
        env.reset(seed=args.seed)
        # 运行时实测 origin 离地高度, 更新 viz_waypoints 投影所需参数
        vw.GROUND_Z_EGO = -float(env.agent.origin.getZ(env.engine.render))
        origin = np.array(env.agent.position[:2], dtype=float)
        lap_counter = LapCounter(origin)
        prev_laps = 0
        sim_t = 0.0
        n_ticks = 0
        print(f"[viewer] 启动: ckpt={args.ckpt} 目标 {args.speed:.0f} km/h "
              f"device={device} render={'3D+panel' if use_render else 'panel-only'} "
              f"{'realtime 10Hz' if args.realtime else 'max-speed'} "
              f"GROUND_Z={vw.GROUND_Z_EGO:.4f}")

        while True:
            tick_start = time.time()
            v = env.agent

            # 1) 抓取前相机 RGB
            bgr = tu.grab_rgb(env, v)
            pil = Image.fromarray(bgr[:, :, ::-1])  # BGR → RGB

            # 2) 模型预测 waypoints (ego 系 6×2)
            wp_ego = predictor.predict(pil, float(v.speed_km_h), float(v.steering))

            # 3) 控制器 → 动作
            action = controller.compute(wp_ego, float(v.speed_km_h), config.SIM_DT)
            env.step(action)
            sim_t += config.SIM_DT
            n_ticks += 1

            # 4) 可视化面板: 相机图 + waypoints 叠加 + 控制量
            panel = vw.overlay(bgr, wp_ego, sim_t, float(v.speed_km_h))
            # 在面板底部叠加控制器输出
            import cv2
            steer_val, tb_val = action[0], action[1]
            ctrl_text = f"steer={steer_val:+.3f}  throttle/brake={tb_val:+.3f}  action=[{steer_val:.3f}, {tb_val:.3f}]"
            cv2.putText(panel, ctrl_text, (8, panel.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            if panel_fig is None:
                panel_fig, panel_ax = plt.subplots(1, 1, figsize=(8, 4.8))
                panel_ax.set_axis_off()
                panel_img = panel_ax.imshow(panel[:, :, ::-1])  # BGR→RGB for matplotlib
                panel_fig.canvas.manager.set_window_title("Waypoints Panel")
                panel_fig.tight_layout(pad=0)
                panel_fig.show()
            else:
                panel_img.set_data(panel[:, :, ::-1])
                panel_fig.canvas.draw_idle()
                panel_fig.canvas.flush_events()

            v = env.agent
            pos = (float(v.position[0]), float(v.position[1]))
            offroad, collision = tu.vehicle_status(env, v)
            lat = tracker.signed_error(*pos)

            laps = lap_counter.update(pos)
            if laps > prev_laps:
                prev_laps = laps
                env.agent.reset_navigation()
                print(f"[viewer] 完成第 {laps} 圈 @ t={sim_t:.1f}s", flush=True)

            if n_ticks % 20 == 0:
                print(f"[viewer] t={sim_t:6.1f}s speed={v.speed_km_h:5.1f} "
                      f"lat={lat:+7.2f}m offroad={int(offroad)} "
                      f"wp0=({wp_ego[0, 0]:.1f},{wp_ego[0, 1]:.1f})", flush=True)

            if collision:
                aborted = "collision"
                print(f"[viewer] 碰撞 @ t={sim_t:.1f}s, 停止", flush=True)
                break
            if args.max_seconds > 0 and sim_t >= args.max_seconds:
                aborted = "max_seconds"
                break
            # 面板被关闭也停止
            if not plt.fignum_exists(panel_fig.number) if panel_fig else False:
                aborted = "user_quit"
                break

            if args.realtime:
                elapsed = time.time() - tick_start
                if elapsed < config.SIM_DT:
                    time.sleep(config.SIM_DT - elapsed)
    except KeyboardInterrupt:
        aborted = "keyboard"
    finally:
        if panel_fig:
            plt.close(panel_fig)
        env.close()

    print(f"[viewer] 结束: ticks={n_ticks} sim_t={sim_t:.1f}s laps={prev_laps} "
          f"abort={aborted}")


def _refline_waypoints(ref_x, ref_y, vehicle, speed_kmh):
    """无相机时的兜底: 从参考线沿弧长推进 6 个 0.5s 间隔点, 转到 ego 系.

    与 validate_controller.py 的 D3 决策一致 (几何 + 实测速度), 这里只做最简版.
    """
    from make_waypoints import world_to_ego

    x, y = float(vehicle.position[0]), float(vehicle.position[1])
    h = float(vehicle.heading_theta)
    step_m = float(speed_kmh) / 3.6 * config.WP_DT
    d2 = (ref_x - x) ** 2 + (ref_y - y) ** 2
    idx = int(np.argmin(d2))
    pts = []
    for k in range(1, config.WP_COUNT + 1):
        j = (idx + int(round(k * step_m))) % len(ref_x)
        pts.append([ref_x[j], ref_y[j]])
    return world_to_ego(np.array(pts), (x, y, h))


if __name__ == "__main__":
    main()
