"""阶段2: 轨迹 → 时间 waypoints 数据集 (未来 3s, 2Hz, 6 点).

核心后处理:
- 对每一帧 i (时刻 t_i), 取 t_i + 0.5, t_i + 1.0, ..., t_i + 3.0 共 WP_COUNT 个时刻,
  在轨迹上按时间线性插值得到世界坐标点;
- 支持单圈与多圈变速轨迹 (5 圈): 多圈沿同一条连续时间轴插值, 不跨圈拼接;
- 圈尾已多录 TAIL_EXTRA_S(6s), 真实未来直接可取; 仅最后不足一个时域的窗口
  按末速度直线延拓 (圈尾是直道, 与原轨迹一致);
- 开环/片段轨迹: 末尾按最后航向+速度匀速外推;
- 世界坐标 → ego 车体系 (x 前, y 左), 与 train/collect_data.py 的约定一致。

输出: output/dataset_waypoints*.npz + *_meta.json

用法:
    python3 make_waypoints.py [--traj PATH] [--out PREFIX]
    python3 make_waypoints.py --traj output/trajectory_5laps.json \\
        --out output/dataset_waypoints_5laps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config
import traj_utils as tu


def world_to_ego(pts_xy: np.ndarray, ego_xy, ego_heading: float) -> np.ndarray:
    """世界系 [N,2] → ego 系 [N,2] (x 前, y 左)."""
    dx = pts_xy[:, 0] - ego_xy[0]
    dy = pts_xy[:, 1] - ego_xy[1]
    c, s = np.cos(ego_heading), np.sin(ego_heading)
    out = np.empty_like(pts_xy)
    out[:, 0] = dx * c + dy * s
    out[:, 1] = -dx * s + dy * c
    return out


def _build_interp_arrays(arr: dict, closed_loop: bool):
    """构造用于未来时刻插值的 (t, x, y) 序列.

    注意: 不再把圈尾"接回圈头"。`lap_time_s` 由终点 40 m 半径触发, 并非严格
    周期 (实测首末点差 ~77 m), 按它平移会让圈尾帧的 GT 错位几十米; 而采集时
    已在圈尾多录 TAIL_EXTRA_S(6 s) > 时域(3 s), 真实未来本就录到了。
    因此只在最后不足一个时域的窗口内, 按末帧速度/航向直线延拓 (该段是直道,
    匀速延拓与原轨迹一致)。
    """
    t, x, y = arr["t"], arr["x"], arr["y"]
    if closed_loop and len(t) >= 2:
        dt_nom = float(np.median(np.diff(t)))
        v = float(arr["speed_kmh"][-1]) / 3.6
        h = float(arr["heading"][-1])
        n_ext = int(np.ceil(config.WP_HORIZON_S / max(dt_nom, 1e-6))) + 1
        dt = np.arange(1, n_ext + 1) * dt_nom
        t = np.concatenate([t, t[-1] + dt])
        x = np.concatenate([x, x[-1] + v * np.cos(h) * dt])
        y = np.concatenate([y, y[-1] + v * np.sin(h) * dt])
    return t, x, y


def generate_waypoints(frames: list, metadata: dict):
    """返回 (wp_ego[N,WP_COUNT,2], wp_world[N,WP_COUNT,2], 使用的帧数组)."""
    arr = tu.frames_to_arrays(frames)
    n = len(arr["t"])
    offsets = np.arange(1, config.WP_COUNT + 1) * config.WP_DT  # 0.5..3.0

    laps = int(metadata.get("laps", 0))
    quick = bool(metadata.get("quick", False))
    multi_lap = laps > 1
    closed_loop = (laps >= 1) and (not quick)
    lap_period = None
    lap_times = [float(x) for x in metadata.get("lap_times_s") or []]
    if closed_loop:
        # 单圈: 沿用 lap_time_s; 多圈: 用各圈时长的中位数作为"典型圈时"
        if multi_lap and lap_times:
            lap_period = float(np.median(lap_times))
        else:
            lap_period = metadata.get("lap_time_s")
            if lap_period is None:
                lap_period = float(metadata["duration_s"]) - config.TAIL_EXTRA_S
            lap_period = float(lap_period)
        if lap_period <= config.WP_HORIZON_S:
            closed_loop = False

    t_ext, x_ext, y_ext = _build_interp_arrays(arr, closed_loop)
    n_tail_ext = int((arr["t"] + config.WP_HORIZON_S > arr["t"][-1]).sum())

    wp_world = np.zeros((n, config.WP_COUNT, 2), dtype=np.float64)
    wp_ego = np.zeros_like(wp_world)
    extrapolated = 0
    for i in range(n):
        future_t = arr["t"][i] + offsets
        if future_t[-1] <= t_ext[-1]:
            px = np.interp(future_t, t_ext, x_ext)
            py = np.interp(future_t, t_ext, y_ext)
        else:
            # 开环片段末尾: 最后航向 + 最后速度 匀速外推
            in_range = future_t <= t_ext[-1]
            px = np.interp(future_t[in_range], t_ext, x_ext)
            py = np.interp(future_t[in_range], t_ext, y_ext)
            px = np.concatenate([px, np.full(int((~in_range).sum()), np.nan)])
            py = np.concatenate([py, np.full(int((~in_range).sum()), np.nan)])
            v_ms = arr["speed_kmh"][i] / 3.6
            h = arr["heading"][i]
            dt_extra = future_t[~in_range] - arr["t"][i]
            px[~in_range] = arr["x"][i] + v_ms * np.cos(h) * dt_extra
            py[~in_range] = arr["y"][i] + v_ms * np.sin(h) * dt_extra
            extrapolated += int((~in_range).sum())
        pts = np.stack([px, py], axis=1)
        wp_world[i] = pts
        wp_ego[i] = world_to_ego(pts, (arr["x"][i], arr["y"][i]), arr["heading"][i])

    info = dict(closed_loop=closed_loop, multi_lap=multi_lap, lap_period=lap_period,
                lap_times_s=lap_times,
                lap_speeds_kmh=[float(x) for x in metadata.get("lap_speeds_kmh") or []],
                extrapolated_points=extrapolated, tail_frames_extended=n_tail_ext)
    return wp_ego, wp_world, arr, info


def build_dataset(traj_path=None, out_prefix=None, verbose: bool = True):
    traj_path = Path(traj_path or (config.OUT_DIR / "trajectory_expert.json"))
    out_prefix = Path(out_prefix or (config.OUT_DIR / "dataset_waypoints"))
    frames, metadata = tu.load_trajectory(traj_path)
    wp_ego, wp_world, arr, info = generate_waypoints(frames, metadata)
    n = len(frames)

    # 训练/验证划分 (仅供参考; training/ 会按圈重新划分):
    # - 多圈: 留最后一圈做验证 (E1: 第 5 圈 72 km/h), 其余训练, lap=0 起始帧不计入
    # - 单圈/片段: 前 90% 训练, 后 10% 验证 (时间顺序)
    if info["multi_lap"]:
        val_lap = int(np.max(arr["lap"]))
        train_mask = (arr["lap"] >= 1) & (arr["lap"] < val_lap)
        val_mask = arr["lap"] == val_lap
    else:
        n_train = int(n * 0.9)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:n_train] = True
        val_mask = ~train_mask

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out_prefix.with_suffix(".npz")
    # 图片相对路径 (采集时未采图的帧为空串), 与轨迹 JSON 同目录解析
    image_path = np.array([f.get("image", "") for f in frames], dtype="<U128")
    np.savez_compressed(
        npz_path,
        wp_ego=wp_ego.astype(np.float32),
        wp_world=wp_world.astype(np.float32),
        t=arr["t"], x=arr["x"], y=arr["y"], heading=arr["heading"],
        speed_kmh=arr["speed_kmh"],
        steering=arr["steering"], throttle=arr["throttle"], brake=arr["brake"],
        lap=arr["lap"], lap_t=arr["lap_t"],
        target_speed_kmh=arr["target_speed_kmh"],
        yaw_rate=arr["yaw_rate"], accel_ms2=arr["accel_ms2"],
        image_path=image_path,
        train_mask=train_mask, val_mask=val_mask,
    )

    # 数据校验: 相邻 waypoint 间距 ≈ v * 0.5s
    seg = np.linalg.norm(np.diff(wp_ego, axis=1), axis=2)  # [N,WP_COUNT-1]
    expected = arr["speed_kmh"][:, None] / 3.6 * config.WP_DT  # [N,1] → 广播 [N,WP_COUNT-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_err = np.abs(seg - expected) / np.maximum(expected, 0.5)
    valid = np.broadcast_to(expected > 0.5, rel_err.shape)  # 低速帧间距校验无意义
    spacing_err = float(np.nanmedian(rel_err[valid])) if valid.any() else 0.0

    # 逐圈诊断: 首点距离 (≈ v×0.5s) 与速度, 用于快速发现圈尾/换速异常
    wp0_dist = np.linalg.norm(wp_ego[:, 0, :], axis=1)
    per_lap = []
    for lap_id in np.unique(arr["lap"]):
        if lap_id <= 0:
            continue
        m = arr["lap"] == lap_id
        per_lap.append({
            "lap": int(lap_id),
            "frames": int(m.sum()),
            "images": int((m & (image_path != "")).sum()),
            "target_speed_kmh": float(np.median(arr["target_speed_kmh"][m])),
            "speed_mean_kmh": round(float(arr["speed_kmh"][m].mean()), 2),
            "wp0_median_m": round(float(np.median(wp0_dist[m])), 3),
            "wp0_max_m": round(float(wp0_dist[m].max()), 3),
        })

    meta = {
        "baseline_id": config.BASELINE_ID,
        "source_trajectory": str(traj_path),
        "num_frames": n,
        "wp_count": config.WP_COUNT,
        "wp_horizon_s": config.WP_HORIZON_S,
        "wp_freq_hz": config.WP_FREQ_HZ,
        "wp_frame": "ego (x forward, y left), meters",
        "record_hz": config.RECORD_HZ,
        "train_frames": int(train_mask.sum()),
        "val_frames": int(val_mask.sum()),
        "num_images": int((image_path != "").sum()),
        "image_interval_s": metadata.get("image_interval_s"),
        "image_dir": metadata.get("image_dir"),
        "image_format": metadata.get("image_format"),
        "camera": dict(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT,
                       fov_deg=config.IMAGE_FOV_DEG) if int((image_path != "").sum()) else None,
        "spacing_median_rel_err": round(spacing_err, 4),
        "per_lap": per_lap,
        **info,
        "trajectory_metadata": metadata,
    }
    meta_path = out_prefix.with_name(out_prefix.name + "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"[waypoints] {n} 帧 × {config.WP_COUNT} 点 (未来{config.WP_HORIZON_S:.0f}s @ "
              f"{config.WP_FREQ_HZ:.0f}Hz), 多圈={info['multi_lap']}, "
              f"闭环延拓={info['closed_loop']}, 外推点数={info['extrapolated_points']}, "
              f"间距校验中位相对误差={spacing_err:.3f}", flush=True)
        for pl in per_lap:
            print(f"[waypoints]   第{pl['lap']}圈: {pl['frames']} 帧 / {pl['images']} 图, "
                  f"目标 {pl['target_speed_kmh']:.0f} km/h (实测均速 "
                  f"{pl['speed_mean_kmh']:.1f}), |wp0| 中位 "
                  f"{pl['wp0_median_m']:.2f} m, max {pl['wp0_max_m']:.2f} m", flush=True)
        print(f"[waypoints] 数据集 → {npz_path}", flush=True)
    return npz_path, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="输出前缀 (不含扩展名)")
    args = parser.parse_args()
    build_dataset(traj_path=args.traj, out_prefix=args.out)