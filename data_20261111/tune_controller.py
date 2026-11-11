"""阶段4: 几何控制器自动调参 — 短片段闭环参数扫描.

在 quick 片段数据集上做小型网格搜索 (前瞻距离 × 横向增益),
评分 = 平均|横向误差| + 出界惩罚; 中止(碰撞/卡死/超时)记为无穷大。
最优参数写入 output/best_geo_params.json, 供整圈评测使用。

用法:
    python3 tune_controller.py [--data NPZ] [--seconds 60]
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

import config
import traj_utils as tu
from controllers import GeometricController
from evaluate import WaypointReplay


def run_segment(params: dict, replay: WaypointReplay, lat_tracker,
                seconds: float, env=None, start_t: float = 0.0, traj_arr=None):
    """在已有 env 上跑一段闭环 (复用 env 避免反复建图), 返回评分与指标.

    start_t > 0 时先把车辆传送到轨迹对应位姿 (用于直接调弯道段)。
    """
    own_env = env is None
    if own_env:
        env = tu.build_headless_env()
    try:
        env.reset(seed=0)
        lat_tracker.reset()  # 每组参数从同一起点开始, 避免里程索引残留
        sim_t = 0.0
        if start_t > 0.0 and traj_arr is not None:
            i = int(np.clip(np.searchsorted(traj_arr["t"], start_t), 0, len(traj_arr["t"]) - 1))
            v = env.agent
            v.set_position([float(traj_arr["x"][i]), float(traj_arr["y"][i])])
            v.set_heading_theta(float(traj_arr["heading"][i]))
            spd = float(traj_arr["speed_kmh"][i]) / 3.6
            h = float(traj_arr["heading"][i])
            v.set_velocity([spd * np.cos(h), spd * np.sin(h)])
            sim_t = float(traj_arr["t"][i])
        end_t = sim_t + seconds
        ctrl = GeometricController(params)
        lats = []
        n_off = 0
        n_col = 0
        low_speed_since = None
        aborted = None
        while sim_t < end_t:
            v = env.agent
            wp = replay.at(sim_t, ego_xy=v.position[:2], ego_heading=v.heading_theta)
            action = ctrl.compute(wp, float(v.speed_km_h), config.SIM_DT)
            env.step(action)
            sim_t += config.SIM_DT
            v = env.agent
            lats.append(lat_tracker.signed_error(float(v.position[0]), float(v.position[1])))
            off, col = tu.vehicle_status(env, v)
            n_off += int(off)
            n_col += int(col)
            if col:
                aborted = "collision"
                break
            if sim_t > start_t + 8.0 and v.speed_km_h < 2.0:
                low_speed_since = sim_t if low_speed_since is None else low_speed_since
                if sim_t - low_speed_since > 4.0:
                    aborted = "stuck"
                    break
            else:
                low_speed_since = None
        n = len(lats)
        lat_mean = float(np.mean(np.abs(lats))) if n else float("inf")
        off_ratio = n_off / n if n else 1.0
        score = lat_mean + 20.0 * off_ratio
        if aborted:
            score = float("inf")
        return dict(score=score, lat_abs_mean_m=lat_mean, offroad_ratio=off_ratio,
                    collision_frames=n_col, aborted=aborted, frames=n)
    finally:
        if own_env:
            env.close()


def tune(data_path=None, seconds: float = 60.0, start_t: float = 0.0, verbose: bool = True):
    data_path = data_path or (config.OUT_DIR / "dataset_waypoints.npz")
    replay = WaypointReplay(data_path)
    traj_path = config.OUT_DIR / "trajectory_expert.json"
    ref_x, ref_y, _ = tu.extract_lap_reference(traj_path)
    lat_tracker = tu.LateralErrorTracker(ref_x, ref_y)
    frames, _ = tu.load_trajectory(traj_path)
    traj_arr = tu.frames_to_arrays(frames)
    # 片段不能超出轨迹覆盖范围 (末尾 5s 的 waypoints 是外推的, 不用来调参)
    max_start = float(traj_arr["t"][-1]) - config.WP_HORIZON_S - seconds
    start_t = float(np.clip(start_t, 0.0, max(max_start, 0.0)))

    grid = list(itertools.product(
        [4.0, 6.0, 9.0],        # look_base
        [0.25, 0.35, 0.5],      # look_scale (前瞻随速系数)
    ))
    env = tu.build_headless_env()
    results = []
    try:
        for look_base, look_scale in grid:
            params = dict(look_base=look_base, look_scale=look_scale)
            m = run_segment(params, replay, lat_tracker, seconds, env=env,
                            start_t=start_t, traj_arr=traj_arr)
            results.append(dict(params=params, **m))
            if verbose:
                print(f"[tune] look_base={look_base:4.1f} look_scale={look_scale:4.2f} → "
                      f"score={m['score']:7.3f} lat={m['lat_abs_mean_m']:.3f}m "
                      f"off={m['offroad_ratio']:.2%} aborted={m['aborted']}", flush=True)
    finally:
        env.close()

    results.sort(key=lambda r: r["score"])
    best = results[0]
    if not np.isfinite(best["score"]):
        raise RuntimeError("所有参数组合都失败, 请检查控制器/数据")
    out_path = config.OUT_DIR / "best_geo_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best["params"], f, indent=2)
    with open(config.OUT_DIR / "tune_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[tune] 调参片段: start_t={start_t:.1f}s, 时长={seconds:.0f}s", flush=True)
        print(f"[tune] 最优参数: {best['params']} (score={best['score']:.3f}) → {out_path}", flush=True)
    return best["params"], results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--start", type=float, default=0.0, help="片段起始仿真时刻 (s)")
    args = parser.parse_args()
    tune(data_path=args.data, seconds=args.seconds, start_t=args.start)