"""阶段2: 整圈轨迹 → 时间 waypoints 数据集 (未来 5s, 2Hz, 10 点).

核心后处理:
- 对每一帧 i (时刻 t_i), 取 t_i + 0.5, t_i + 1.0, ..., t_i + 5.0 共 10 个时刻,
  在轨迹上按时间线性插值得到世界坐标点;
- 闭环整圈轨迹: 按圈周期延拓 (圈尾接圈头), 保证每帧都有完整未来 5s;
- 开环/片段轨迹: 末尾按最后航向+速度匀速外推;
- 世界坐标 → ego 车体系 (x 前, y 左), 与 train/collect_data.py 的约定一致。

输出: output/dataset_waypoints.npz + dataset_meta.json

用法:
    python3 make_waypoints.py [--traj PATH] [--out PREFIX]
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


def _build_interp_arrays(arr: dict, closed_loop: bool, lap_period: float | None):
    """构造用于未来时刻插值的 (t, x, y) 序列, 闭环时做周期延拓."""
    t, x, y = arr["t"], arr["x"], arr["y"]
    if closed_loop and lap_period is not None and lap_period > 0:
        # 圈尾之后接圈头 (位置平移一个周期不需要, 闭环位置本身重合),
        # 延拓两圈以保证最后帧 +5s 仍在插值范围内
        t_ext = np.concatenate([t, t + lap_period, t + 2 * lap_period])
        x_ext = np.concatenate([x, x, x])
        y_ext = np.concatenate([y, y, y])
        return t_ext, x_ext, y_ext
    return t, x, y


def generate_waypoints(frames: list, metadata: dict):
    """返回 (wp_ego[N,10,2], wp_world[N,10,2], 使用的帧数组)."""
    arr = tu.frames_to_arrays(frames)
    n = len(arr["t"])
    offsets = np.arange(1, config.WP_COUNT + 1) * config.WP_DT  # 0.5..5.0

    laps = int(metadata.get("laps", 0))
    quick = bool(metadata.get("quick", False))
    closed_loop = (laps >= 1) and (not quick)
    lap_period = None
    if closed_loop:
        # 优先用采集时记录的精确完圈时刻作为周期
        lap_period = metadata.get("lap_time_s")
        if lap_period is None:
            lap_period = float(metadata["duration_s"]) - config.TAIL_EXTRA_S
        lap_period = float(lap_period)
        if lap_period <= config.WP_HORIZON_S:
            closed_loop = False

    t_ext, x_ext, y_ext = _build_interp_arrays(arr, closed_loop, lap_period)

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

    info = dict(closed_loop=closed_loop, lap_period=lap_period,
                extrapolated_points=extrapolated)
    return wp_ego, wp_world, arr, info


def build_dataset(traj_path=None, out_prefix=None, verbose: bool = True):
    traj_path = Path(traj_path or (config.OUT_DIR / "trajectory_expert.json"))
    out_prefix = Path(out_prefix or (config.OUT_DIR / "dataset_waypoints"))
    frames, metadata = tu.load_trajectory(traj_path)
    wp_ego, wp_world, arr, info = generate_waypoints(frames, metadata)
    n = len(frames)

    # 训练/验证划分: 前 90% 训练, 后 10% 验证 (时间顺序)
    n_train = int(n * 0.9)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:n_train] = True

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out_prefix.with_suffix(".npz")
    # 图片相对路径 (采集时未采图的帧为空串), 与轨迹 JSON 同目录解析
    image_path = np.array([f.get("image", "") for f in frames], dtype="<U64")
    np.savez_compressed(
        npz_path,
        wp_ego=wp_ego.astype(np.float32),
        wp_world=wp_world.astype(np.float32),
        t=arr["t"], x=arr["x"], y=arr["y"], heading=arr["heading"],
        speed_kmh=arr["speed_kmh"],
        steering=arr["steering"], throttle=arr["throttle"], brake=arr["brake"],
        image_path=image_path,
        train_mask=train_mask, val_mask=~train_mask,
    )

    # 数据校验: 相邻 waypoint 间距 ≈ v * 0.5s
    seg = np.linalg.norm(np.diff(wp_ego, axis=1), axis=2)  # [N,9]
    expected = arr["speed_kmh"][:, None] / 3.6 * config.WP_DT  # [N,1] → 广播 [N,9]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_err = np.abs(seg - expected) / np.maximum(expected, 0.5)
    valid = np.broadcast_to(expected > 0.5, rel_err.shape)  # 低速帧间距校验无意义
    spacing_err = float(np.nanmedian(rel_err[valid])) if valid.any() else 0.0

    meta = {
        "source_trajectory": str(traj_path),
        "num_frames": n,
        "wp_count": config.WP_COUNT,
        "wp_horizon_s": config.WP_HORIZON_S,
        "wp_freq_hz": config.WP_FREQ_HZ,
        "wp_frame": "ego (x forward, y left), meters",
        "record_hz": config.RECORD_HZ,
        "train_frames": n_train,
        "val_frames": n - n_train,
        "num_images": int((image_path != "").sum()),
        "image_interval_s": metadata.get("image_interval_s"),
        "spacing_median_rel_err": round(spacing_err, 4),
        **info,
        "trajectory_metadata": metadata,
    }
    meta_path = out_prefix.with_name(out_prefix.name + "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"[waypoints] {n} 帧 × {config.WP_COUNT} 点 (未来{config.WP_HORIZON_S:.0f}s @ "
              f"{config.WP_FREQ_HZ:.0f}Hz), 闭环延拓={info['closed_loop']}, "
              f"外推点数={info['extrapolated_points']}, 间距校验中位相对误差={spacing_err:.3f}", flush=True)
        print(f"[waypoints] 数据集 → {npz_path}", flush=True)
    return npz_path, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="输出前缀 (不含扩展名)")
    args = parser.parse_args()
    build_dataset(traj_path=args.traj, out_prefix=args.out)