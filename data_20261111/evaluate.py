"""阶段5: 无头闭环评测 — 控制器消费 waypoints, 实际驾驶一整圈.

waypoints 来源: 专家整圈数据集按仿真时刻回放 (等价于"完美模型预测"),
用于单独验证控制器本身; 后续接真实模型时只需替换 waypoints 来源。

用法:
    python3 evaluate.py --controller geometric [--params JSON] [--quick]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import config
import traj_utils as tu
from controllers import GeometricController


class WaypointReplay:
    """按仿真时刻回放未来路径, 等价于"完美预测模型".

    关键: 回放世界系 wp_world, 再用车辆【当前实际位姿】转到 ego 系。
    不能直接回放 wp_ego —— 那是专家位姿下的 ego 系, 评测车一旦偏离专家位姿,
    坐标基准就错了, 偏差会滚雪球 (实测 90s 漂出 70m)。
    """

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.t = data["t"]
        self.wp_world = data["wp_world"]

    def at(self, sim_t: float, ego_xy=None, ego_heading: float | None = None) -> np.ndarray:
        idx = int(np.searchsorted(self.t, sim_t, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self.t) - 1))
        wp = self.wp_world[idx].astype(float)
        if ego_xy is None or ego_heading is None:
            raise ValueError("WaypointReplay.at 需要车辆当前位姿 (ego_xy, ego_heading)")
        from make_waypoints import world_to_ego

        return world_to_ego(wp, ego_xy, float(ego_heading))


def make_controller(name: str, params: dict | None = None):
    if name == "geometric":
        return GeometricController(params)
    raise ValueError(f"未知控制器: {name}")


def run_closed_loop(controller_name: str, params: dict | None = None,
                    quick: bool = False, tag: str | None = None,
                    npz_path=None, verbose: bool = True):
    from simple_track import LapCounter

    npz_path = Path(npz_path or (config.OUT_DIR / "dataset_waypoints.npz"))
    tag = tag or controller_name
    out_dir = config.OUT_DIR / f"eval_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    replay = WaypointReplay(npz_path)
    controller = make_controller(controller_name, params)
    # 基准 = 专家轨迹提取的单圈路径 (参考 JSON 含多圈重叠, 不适合做误差基准)
    ref_x, ref_y, _ = tu.extract_lap_reference(config.OUT_DIR / "trajectory_expert.json")
    lat_tracker = tu.LateralErrorTracker(ref_x, ref_y)

    env = tu.build_headless_env()
    log = []
    wall_start = time.time()
    try:
        env.reset(seed=0)
        controller.reset()
        origin = np.array(env.agent.position[:2], dtype=float)
        lap_counter = LapCounter(origin)
        sim_t = 0.0
        laps = 0
        max_sim_t = config.QUICK_SECONDS if quick else 1500.0
        low_speed_since = None
        aborted = None

        if verbose:
            print(f"[eval:{tag}] 开始无头闭环 ({'quick片段' if quick else '整圈'}), "
                  f"控制器={controller_name}", flush=True)

        while True:
            v = env.agent
            wp = replay.at(sim_t, ego_xy=v.position[:2], ego_heading=v.heading_theta)
            action = controller.compute(wp, float(v.speed_km_h), config.SIM_DT)
            env.step(action)
            sim_t += config.SIM_DT

            v = env.agent
            x, y = float(v.position[0]), float(v.position[1])
            lat = lat_tracker.signed_error(x, y)
            offroad, collision = tu.vehicle_status(env, v)
            log.append(dict(
                t=round(sim_t, 3), x=x, y=y,
                heading=float(v.heading_theta),
                speed_kmh=float(v.speed_km_h),
                steering=float(action[0]), throttle_brake=float(action[1]),
                lat_err=lat, offroad=offroad, collision=collision,
            ))

            laps = lap_counter.update(v.position)
            if laps >= 1 and not quick:
                break
            if quick and sim_t >= max_sim_t:
                break
            if sim_t >= max_sim_t:
                aborted = "timeout"
                break
            if collision:
                aborted = "collision"
                break
            # 卡死检测: 起步 10s 后持续 5s 速度 < 2km/h
            if sim_t > 10.0 and v.speed_km_h < 2.0:
                low_speed_since = sim_t if low_speed_since is None else low_speed_since
                if sim_t - low_speed_since > 5.0:
                    aborted = "stuck"
                    break
            else:
                low_speed_since = None

            if verbose and len(log) % 1000 == 0:
                print(f"[eval:{tag}] t={sim_t:7.1f}s speed={v.speed_km_h:5.1f}km/h "
                      f"lat={lat:+5.2f}m laps={laps}", flush=True)

        metrics = _summarize(log, laps, quick, aborted, controller_name, params)
        metrics["wall_time_s"] = round(time.time() - wall_start, 1)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        _save_log_csv(out_dir / "drive_log.csv", log)
        _plot(out_dir, log, ref_x, ref_y, tag, metrics)
        if verbose:
            print(f"[eval:{tag}] 完成: laps={laps} aborted={aborted} "
                  f"平均|横向|={metrics['lat_abs_mean_m']:.2f}m "
                  f"最大|横向|={metrics['lat_abs_max_m']:.2f}m "
                  f"出界率={metrics['offroad_ratio']:.3%} 碰撞帧={metrics['collision_frames']} "
                  f"耗时 {metrics['wall_time_s']}s → {out_dir}", flush=True)
        return metrics, out_dir
    finally:
        env.close()


def _summarize(log, laps, quick, aborted, controller_name, params):
    lat = np.array([r["lat_err"] for r in log])
    speed = np.array([r["speed_kmh"] for r in log])
    steer = np.array([r["steering"] for r in log])
    n = len(log)
    return dict(
        controller=controller_name,
        params=params or {},
        quick=bool(quick),
        completed=bool(laps >= 1) if not quick else aborted is None,
        laps=int(laps),
        aborted=aborted,
        duration_s=round(log[-1]["t"], 2) if n else 0.0,
        num_frames=n,
        lat_abs_mean_m=float(np.mean(np.abs(lat))) if n else None,
        lat_abs_p95_m=float(np.percentile(np.abs(lat), 95)) if n else None,
        lat_abs_max_m=float(np.max(np.abs(lat))) if n else None,
        offroad_frames=int(sum(r["offroad"] for r in log)),
        offroad_ratio=float(np.mean([r["offroad"] for r in log])) if n else None,
        collision_frames=int(sum(r["collision"] for r in log)),
        speed_mean_kmh=float(np.mean(speed)) if n else None,
        speed_min_kmh=float(np.min(speed)) if n else None,
        steer_abs_mean=float(np.mean(np.abs(steer))) if n else None,
        steer_rate_abs_mean=float(np.mean(np.abs(np.diff(steer)))) if n > 1 else None,
    )


def _save_log_csv(path, log):
    import csv

    if not log:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)


def _plot(out_dir, log, ref_x, ref_y, tag, metrics):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not log:
        return
    step = max(1, len(ref_x) // 4000)
    xs = [r["x"] for r in log]
    ys = [r["y"] for r in log]
    ts = [r["t"] for r in log]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax = axes[0][0]
    ax.plot(ref_x[::step], ref_y[::step], "b-", lw=0.6, alpha=0.5, label="reference")
    ax.plot(xs, ys, "r-", lw=0.9, alpha=0.8, label=f"driven ({tag})")
    ax.set_aspect("equal")
    ax.set_title(f"Trajectory — {tag}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(ts, [r["lat_err"] for r in log], lw=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"Lateral error (mean|.|={metrics['lat_abs_mean_m']:.2f} m)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("m")
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(ts, [r["speed_kmh"] for r in log], lw=0.7, label="speed")
    ax.axhline(config.TARGET_SPEED_KMH, color="g", ls="--", lw=0.8, label="target")
    ax.set_title("Speed")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("km/h")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(ts, [r["steering"] for r in log], lw=0.6, label="steering")
    ax.plot(ts, [r["throttle_brake"] for r in log], lw=0.6, alpha=0.7, label="throttle/brake")
    ax.set_title("Control outputs")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Closed-loop evaluation: {tag} | completed={metrics['completed']} "
                 f"offroad={metrics['offroad_ratio']:.2%}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "eval_plots.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["geometric"], default="geometric")
    parser.add_argument("--params", type=str, default=None, help="几何控制器参数 JSON 文件")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()
    params = None
    if args.params:
        with open(args.params, "r", encoding="utf-8") as f:
            params = json.load(f)
    run_closed_loop(args.controller, params=params, quick=args.quick,
                    tag=args.tag, npz_path=args.data)