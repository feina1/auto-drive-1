#!/usr/bin/env python3
"""绘制录制帧的 X/Y 位置随时间变化曲线

用法:
    python plot_xy.py                                          # 自动找最新录制
    python plot_xy.py records_pid/2026-09-13_18-22             # 指定会话
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent


def find_latest_session():
    root = PROJECT_DIR / "records_pid"
    sessions = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions[0]


def load_frames(session_dir):
    frames_dir = session_dir / "frames"
    meta_path = session_dir / "metadata.json"
    hz = 10
    if meta_path.exists():
        hz = json.loads(meta_path.read_text()).get("record_hz", 10)

    xs, ys, ts = [], [], []
    for f in sorted(frames_dir.glob("*.json")):
        d = json.loads(f.read_text())
        xs.append(d["position"]["x"])
        ys.append(d["position"]["y"])
        ts.append((d["frame_id"] - 1) / hz)
    return ts, xs, ys


def main():
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        session_dir = Path(sys.argv[1]).resolve()
    else:
        session_dir = find_latest_session()

    print(f"会话: {session_dir.name}")
    ts, xs, ys = load_frames(session_dir)
    print(f"帧数: {len(ts)} | 时长: {ts[-1]:.1f}s")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(ts, xs, linewidth=0.8)
    ax1.set_ylabel("X (m)")
    ax1.set_title(f"{session_dir.name}  —  X / Y vs Time")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ts, ys, linewidth=0.8, color="tab:orange")
    ax2.set_ylabel("Y (m)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(session_dir / "xy_time.png", dpi=150)
    print(f"已保存: {session_dir / 'xy_time.png'}")
    plt.show()


if __name__ == "__main__":
    main()
