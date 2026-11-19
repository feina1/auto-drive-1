"""阶段1: 专家无头跑圈, 采集整圈状态轨迹; 可选采相机图片 (--images).

专家 = 既有 PathFollower (Pure Pursuit + PID, 参考内侧车道中心线)。

两种用法:
- 单圈 (默认): 70 km/h 跑 1 圈 → output/trajectory_expert.json
- 多圈变速训练集: 每圈一个巡航速度, 例如
      python3 collect_trajectory.py --laps 5 --speeds 40,48,56,64,72 --images \
          --out output/trajectory_5laps.json
  遥测按 10 Hz 全量记录 (位姿/速度/横摆角速度/加速度/控制量/出界/碰撞/圈号),
  图片默认 5 Hz, 保存到 output/images_<out_stem>/ (与单圈图片目录隔离)。

输出 JSON 字段与 record.py 兼容, 另含 t / throttle_brake / lap / lap_t /
target_speed_kmh / yaw_rate / accel_ms2 / velocity / image。
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config
import traj_utils as tu


def _parse_speeds(text):
    if not text:
        return None
    return [float(s) for s in str(text).replace(" ", "").split(",") if s]


def collect(
    out_path=None,
    quick: bool = False,
    verbose: bool = True,
    capture_images: bool = False,
    laps: int = 1,
    lap_speeds=None,
    image_interval_s=None,
    image_dir_name=None,
    image_format=None,
):
    # pid_demo_autonomous 顶层设置 matplotlib 后端, 先强制 Agg 防止弹窗
    import matplotlib

    matplotlib.use("Agg", force=True)
    from pid_demo_autonomous import PathFollower
    from simple_track import LapCounter

    target_laps = 1 if quick else max(1, int(laps))
    if lap_speeds is None:
        lap_speeds = list(config.LAP_SPEEDS_KMH[:target_laps]) if target_laps > 1 \
            else [config.TARGET_SPEED_KMH]
    lap_speeds = [float(s) for s in lap_speeds][:target_laps]
    if len(lap_speeds) < target_laps:  # 速度列表不够时沿用最后一个
        lap_speeds += [lap_speeds[-1]] * (target_laps - len(lap_speeds))

    out_path = Path(out_path or (config.MULTI_LAP_TRAJ if target_laps > 1
                                else config.OUT_DIR / "trajectory_expert.json"))
    ref_x, ref_y = tu.load_reference_path()

    env = tu.build_headless_env(with_camera=capture_images,
                                spawn_speed_kmh=lap_speeds[0])
    traj_dir = out_path.parent

    image_interval = float(image_interval_s if image_interval_s is not None
                           else config.IMAGE_INTERVAL_S)
    image_every = max(1, int(round(image_interval / config.SIM_DT)))
    image_fmt = str(image_format or config.IMAGE_FORMAT).lower().lstrip(".")
    if image_dir_name is None:
        image_dir_name = (config.IMAGE_DIR_NAME if target_laps == 1
                          else f"images_{out_path.stem}")
    image_dir_name = image_dir_name.strip("/")

    frames = []
    wall_start = time.time()
    try:
        env.reset(seed=0)
        follower = PathFollower(ref_x, ref_y, target_speed_kmh=lap_speeds[0])
        origin = np.array(env.agent.position[:2], dtype=float)
        lap_counter = LapCounter(origin)
        prev_laps = 0
        cur_lap = 1
        sim_t = 0.0
        lap_start_t = 0.0
        lap_times = []
        lap_frames = []
        max_sim_t = config.QUICK_SECONDS if quick else 1300.0 * target_laps + config.TAIL_EXTRA_S
        tail_until = None
        n_offroad = 0
        n_collision = 0
        n_images = 0
        aborted = None
        prev_h = float(env.agent.heading_theta)
        prev_speed = float(env.agent.speed_km_h)

        if verbose:
            mode = f"quick {config.QUICK_SECONDS:.0f}s 片段" if quick else f"{target_laps} 圈变速"
            img_note = (f", 每 {image_interval:.1f}s 采一帧{image_fmt.upper()}图片"
                        if capture_images else "")
            speed_note = ", ".join(f"{s:.0f}" for s in lap_speeds)
            print(f"[collect] 开始无头采集 ({mode}), 逐圈目标速度 [{speed_note}] km/h{img_note}",
                  flush=True)

        while True:
            v = env.agent
            action = follower.compute_control(
                float(v.position[0]), float(v.position[1]),
                float(v.heading_theta), float(v.speed_km_h), config.SIM_DT,
            )
            env.step(action)
            sim_t += config.SIM_DT

            v = env.agent
            steering, throttle, brake = tu.split_control(v)
            offroad, collision = tu.vehicle_status(env, v)
            n_offroad += int(offroad)
            n_collision += int(collision)

            heading = float(v.heading_theta)
            speed_kmh = float(v.speed_km_h)
            yaw_rate = float(tu.wrap_to_pi(heading - prev_h) / config.SIM_DT)
            accel = float((speed_kmh - prev_speed) / 3.6 / config.SIM_DT)
            prev_h, prev_speed = heading, speed_kmh
            try:
                vel = np.asarray(v.velocity, dtype=float).reshape(-1)[:2]
            except Exception:
                vel = np.zeros(2)

            frames.append({
                "frame_id": len(frames) + 1,
                "t": round(sim_t, 4),
                "position": {"x": float(v.position[0]), "y": float(v.position[1])},
                "heading": heading,
                "speed_kmh": speed_kmh,
                "velocity": {"x": float(vel[0]), "y": float(vel[1])},
                "yaw_rate": round(yaw_rate, 5),
                "accel_ms2": round(accel, 4),
                "control": {"steering": steering, "throttle": throttle, "brake": brake},
                "throttle_brake": float(v.throttle_brake),
                "lap": int(cur_lap),
                "lap_t": round(sim_t - lap_start_t, 4),
                "target_speed_kmh": round(float(lap_speeds[cur_lap - 1]), 2),
                "offroad": offroad,
                "collision": collision,
            })

            # 按固定仿真间隔采图 (帧号对齐, 便于和 npz 的 image_path 对上)
            if capture_images and frames[-1]["frame_id"] % image_every == 0:
                rel = f"{image_dir_name}/frame_{frames[-1]['frame_id']:07d}.{image_fmt}"
                tu.capture_rgb(env, v, traj_dir / rel)
                frames[-1]["image"] = rel
                n_images += 1

            laps = lap_counter.update(v.position)
            if laps > prev_laps:
                prev_laps = laps
                lap_times.append(round(sim_t - lap_start_t, 2))
                lap_frames.append(len(frames))
                lap_start_t = sim_t
                env.agent.reset_navigation()
                if prev_laps >= target_laps:
                    # 终点后再录 TAIL_EXTRA_S, 保证末尾帧有完整未来 3s
                    tail_until = sim_t + config.TAIL_EXTRA_S
                    if verbose:
                        print(f"[collect] 完成第 {prev_laps}/{target_laps} 圈 @ t={sim_t:.1f}s, "
                              f"再录 {config.TAIL_EXTRA_S:.0f}s 收尾", flush=True)
                else:
                    cur_lap = prev_laps + 1
                    follower.set_target_speed(lap_speeds[cur_lap - 1])
                    if verbose:
                        print(f"[collect] 第 {prev_laps} 圈完成 @ t={sim_t:.1f}s "
                              f"({lap_times[-1]:.1f}s), 下一圈目标 "
                              f"{lap_speeds[cur_lap - 1]:.0f} km/h", flush=True)

            if verbose and len(frames) % 2000 == 0:
                print(f"[collect] t={sim_t:7.1f}s frames={len(frames)} "
                      f"speed={speed_kmh:5.1f}km/h lap={cur_lap} laps={laps}", flush=True)

            if quick and sim_t >= max_sim_t:
                break
            if tail_until is not None and sim_t >= tail_until:
                break
            if collision:
                aborted = "collision"
                print(f"[collect] 碰撞 @ t={sim_t:.1f}s, 停止并保留已采数据", flush=True)
                break
            if sim_t >= max_sim_t:
                aborted = "timeout"
                print(f"[collect] 超时 {max_sim_t:.0f}s, 已采到第 {prev_laps} 圈, 停止", flush=True)
                break

        metadata = {
            "collect_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "PathFollower expert (pid_demo_autonomous)",
            "map": "SimpleTrackMap",
            "record_hz": config.RECORD_HZ,
            "sim_dt": config.SIM_DT,
            "target_speed_kmh": lap_speeds[0] if target_laps == 1 else list(lap_speeds),
            "lap_speeds_kmh": list(lap_speeds[:max(prev_laps, 1)]),
            "planned_laps": target_laps,
            "quick": bool(quick),
            "laps": int(prev_laps),
            "lap_times_s": lap_times,
            "lap_frames": lap_frames,
            # 单圈兼容字段: 取第一圈完圈时刻; 多圈时 make_waypoints 只用 lap_times_s
            "lap_time_s": lap_times[0] if lap_times else None,
            "num_frames": len(frames),
            "duration_s": round(sim_t, 2),
            "offroad_frames": n_offroad,
            "collision_frames": n_collision,
            "aborted": aborted,
            "num_images": n_images,
            "image_interval_s": image_interval if capture_images else None,
            "image_format": image_fmt if capture_images else None,
            "image_dir": image_dir_name if capture_images else None,
            "camera": dict(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT,
                           fov_deg=config.IMAGE_FOV_DEG) if capture_images else None,
            "wall_time_s": round(time.time() - wall_start, 1),
        }
        tu.save_trajectory(out_path, frames, metadata)
        if verbose:
            print(f"[collect] 完成: {len(frames)} 帧, {sim_t:.1f}s, {prev_laps} 圈, "
                  f"出界帧={n_offroad}, 碰撞帧={n_collision}, 图片={n_images} 张, "
                  f"耗时 {metadata['wall_time_s']}s → {out_path}", flush=True)
        return out_path, metadata
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="只采前 90s 片段")
    parser.add_argument("--laps", type=int, default=1, help="采集圈数 (默认 1)")
    parser.add_argument("--speeds", type=str, default=None,
                        help="逐圈目标速度 km/h, 逗号分隔, 如 40,48,56,64,72")
    parser.add_argument("--images", action="store_true",
                        help=f"采相机图片, 默认每 {config.IMAGE_INTERVAL_S:.1f}s 一帧 (离屏)")
    parser.add_argument("--image-interval", type=float, default=None,
                        help="采图间隔秒 (默认 config.IMAGE_INTERVAL_S)")
    parser.add_argument("--image-format", type=str, default=None, choices=["jpg", "png"])
    parser.add_argument("--image-dir", type=str, default=None,
                        help="图片相对 output/ 的子目录名")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    collect(
        out_path=args.out,
        quick=args.quick,
        capture_images=args.images,
        laps=args.laps,
        lap_speeds=_parse_speeds(args.speeds),
        image_interval_s=args.image_interval,
        image_dir_name=args.image_dir,
        image_format=args.image_format,
    )