"""阶段1: 专家无头跑圈, 采集整圈状态轨迹; 可选每 2s 采一帧相机图片 (--images).

专家 = 既有 PathFollower (Pure Pursuit + PID, 参考内侧车道中心线)。
输出: output/trajectory_expert.json (字段与 record.py 兼容, 另含 t / throttle_brake)。
图片: output/images/frame_{frame_id}.png, 采到图的帧在 JSON 里多一个 "image" 相对路径字段。

用法:
    python3 collect_trajectory.py [--quick] [--images] [--out PATH]
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config
import traj_utils as tu


def collect(out_path=None, quick: bool = False, verbose: bool = True, capture_images: bool = False):
    # pid_demo_autonomous 顶层设置 matplotlib 后端, 先强制 Agg 防止弹窗
    import matplotlib

    matplotlib.use("Agg", force=True)
    from pid_demo_autonomous import PathFollower
    from simple_track import LapCounter

    out_path = out_path or (config.OUT_DIR / "trajectory_expert.json")
    ref_x, ref_y = tu.load_reference_path()

    env = tu.build_headless_env(with_camera=capture_images)
    traj_dir = Path(out_path).parent
    frames = []
    wall_start = time.time()
    try:
        env.reset(seed=0)
        follower = PathFollower(ref_x, ref_y)
        origin = np.array(env.agent.position[:2], dtype=float)
        lap_counter = LapCounter(origin)
        prev_laps = 0
        sim_t = 0.0
        max_sim_t = config.QUICK_SECONDS if quick else 1800.0
        tail_until = None
        lap_time_s = None
        n_offroad = 0
        n_collision = 0
        n_images = 0

        if verbose:
            mode = f"quick {config.QUICK_SECONDS:.0f}s 片段" if quick else "整圈"
            img_note = f", 每 {config.IMAGE_INTERVAL_S:.0f}s 采一帧图片" if capture_images else ""
            print(f"[collect] 开始无头采集 ({mode}), 目标速度 {config.TARGET_SPEED_KMH:.0f} km/h{img_note}", flush=True)

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
            frames.append({
                "frame_id": len(frames) + 1,
                "t": round(sim_t, 4),
                "position": {"x": float(v.position[0]), "y": float(v.position[1])},
                "heading": float(v.heading_theta),
                "speed_kmh": float(v.speed_km_h),
                "control": {"steering": steering, "throttle": throttle, "brake": brake},
                "throttle_brake": float(v.throttle_brake),
                "offroad": offroad,
                "collision": collision,
            })

            # 每 IMAGE_EVERY_STEPS 步 (2s) 在当前帧位姿采一张图片
            if capture_images and frames[-1]["frame_id"] % config.IMAGE_EVERY_STEPS == 0:
                rel = f"{config.IMAGE_DIR_NAME}/frame_{frames[-1]['frame_id']:07d}.png"
                tu.capture_rgb(env, v, traj_dir / rel)
                frames[-1]["image"] = rel
                n_images += 1

            laps = lap_counter.update(v.position)
            if laps > prev_laps:
                prev_laps = laps
                env.agent.reset_navigation()
                if not quick:
                    # 过终点后再录 TAIL_EXTRA_S, 保证末尾帧有完整未来 5s
                    tail_until = sim_t + config.TAIL_EXTRA_S
                    if lap_time_s is None:
                        lap_time_s = sim_t
                    if verbose:
                        print(f"[collect] 完成第 {laps} 圈 @ t={sim_t:.1f}s, 再录 {config.TAIL_EXTRA_S:.0f}s 收尾", flush=True)

            if verbose and len(frames) % 1000 == 0:
                print(f"[collect] t={sim_t:7.1f}s frames={len(frames)} "
                      f"speed={v.speed_km_h:5.1f}km/h laps={laps}", flush=True)

            if quick and sim_t >= max_sim_t:
                break
            if tail_until is not None and sim_t >= tail_until:
                break
            if sim_t >= max_sim_t:
                raise RuntimeError(f"采集超时 ({max_sim_t}s), 未能完成整圈")
            if collision:
                raise RuntimeError(f"专家采集发生碰撞 @ t={sim_t:.1f}s, 中止")

        metadata = {
            "collect_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "PathFollower expert (pid_demo_autonomous)",
            "map": "SimpleTrackMap",
            "record_hz": config.RECORD_HZ,
            "sim_dt": config.SIM_DT,
            "target_speed_kmh": config.TARGET_SPEED_KMH,
            "quick": bool(quick),
            "laps": prev_laps,
            "lap_time_s": round(lap_time_s, 2) if lap_time_s is not None else None,
            "num_frames": len(frames),
            "duration_s": round(sim_t, 2),
            "offroad_frames": n_offroad,
            "collision_frames": n_collision,
            "num_images": n_images,
            "image_interval_s": config.IMAGE_INTERVAL_S if capture_images else None,
            "wall_time_s": round(time.time() - wall_start, 1),
        }
        tu.save_trajectory(out_path, frames, metadata)
        if verbose:
            print(f"[collect] 完成: {len(frames)} 帧, {sim_t:.1f}s, "
                  f"出界帧={n_offroad}, 碰撞帧={n_collision}, 图片={n_images} 张, "
                  f"耗时 {metadata['wall_time_s']}s → {out_path}", flush=True)
        return out_path, metadata
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="只采前 90s 片段")
    parser.add_argument("--images", action="store_true", help=f"每 {config.IMAGE_INTERVAL_S:.0f}s 采一帧相机图片 (离屏)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    collect(out_path=args.out, quick=args.quick, capture_images=args.images)