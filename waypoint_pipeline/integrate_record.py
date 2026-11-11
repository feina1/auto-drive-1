"""预留集成接口: record.py 录制会话 → 本管线轨迹格式.

今天不改动 record.py; 后续合并时, 录完一圈后调用:

    from waypoint_pipeline.integrate_record import session_to_trajectory
    traj_path = session_to_trajectory("records/2026-11-11_20-00")
    # 然后: python3 waypoint_pipeline/make_waypoints.py --traj <traj_path>

说明:
- record.py 的 frames/*.json 为 10 Hz, 字段与本管线兼容 (缺 t, 按 frame_id 推算);
- 会话 metadata 无圈数信息: 若录制时长覆盖一整圈, 可传 laps=1 与 lap_time_s
  启用闭环周期延拓; 否则按开环片段处理 (末尾匀速外推)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
import traj_utils as tu


def session_to_trajectory(session_dir, out_path=None, laps: int = 0,
                          lap_time_s: float | None = None):
    session_dir = Path(session_dir)
    frames_dir = session_dir / "frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"找不到 frames 目录: {frames_dir}")

    meta_path = session_dir / "metadata.json"
    session_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    record_hz = float(session_meta.get("record_hz", config.RECORD_HZ))

    frames = []
    for fp in sorted(frames_dir.glob("*.json")):
        f = json.loads(fp.read_text(encoding="utf-8"))
        f["t"] = round((int(f["frame_id"]) - 1) / record_hz, 4)
        frames.append(f)
    if not frames:
        raise RuntimeError(f"会话无帧数据: {session_dir}")
    frames.sort(key=lambda f: f["frame_id"])

    out_path = Path(out_path or (config.OUT_DIR / "trajectory_from_record.json"))
    metadata = {
        "source": f"record.py session: {session_dir.name}",
        "record_hz": record_hz,
        "sim_dt": 1.0 / record_hz,
        "quick": False,
        "laps": int(laps),
        "lap_time_s": lap_time_s,
        "num_frames": len(frames),
        "duration_s": round(frames[-1]["t"] + 1.0 / record_hz, 2),
        "offroad_frames": sum(int(f.get("offroad", False)) for f in frames),
        "collision_frames": sum(int(f.get("collision", False)) for f in frames),
        "session_metadata": session_meta,
    }
    tu.save_trajectory(out_path, frames, metadata)
    print(f"[integrate] {len(frames)} 帧 → {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=str)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--laps", type=int, default=0)
    parser.add_argument("--lap-time", type=float, default=None)
    args = parser.parse_args()
    session_to_trajectory(args.session_dir, args.out, laps=args.laps, lap_time_s=args.lap_time)