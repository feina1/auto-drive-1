#!/usr/bin/env python3
"""分析参考路径的弯道分布."""
import json
import numpy as np

REF_FILE = "/home/wl/下载/wlwork/auto-drive-1/simple_track_reference_clean.json"

with open(REF_FILE) as f:
    data = json.load(f)

x = np.array(data["reference_path"]["x"])
y = np.array(data["reference_path"]["y"])
N = len(x)
print(f"参考路径: {N} 个点")

# 计算曲率 (滑动窗口)
W = 10
curvatures = []
for i in range(W, N - W):
    dx1, dy1 = x[i] - x[i - W], y[i] - y[i - W]
    dx2, dy2 = x[i + W] - x[i], y[i + W] - y[i]
    cross = dx1 * dy2 - dy1 * dx2
    mag = np.sqrt(dx1**2 + dy1**2) * np.sqrt(dx2**2 + dy2**2) + 1e-10
    curvatures.append(cross / mag)
curvatures = np.array(curvatures)

left_turn = int(np.sum(curvatures > 0.05))
right_turn = int(np.sum(curvatures < -0.05))
straight = int(np.sum(np.abs(curvatures) <= 0.05))
total = len(curvatures)
print(f"左弯: {left_turn} ({100*left_turn/total:.1f}%)")
print(f"右弯: {right_turn} ({100*right_turn/total:.1f}%)")
print(f"直行: {straight} ({100*straight/total:.1f}%)")
print(f"曲率范围: [{curvatures.min():.3f}, {curvatures.max():.3f}]")

# 找连续弯道区间
def find_segments(mask, min_len=20):
    segs = []
    start = None
    for i in range(len(mask)):
        if mask[i] and start is None:
            start = i
        elif not mask[i] and start is not None:
            if i - start >= min_len:
                segs.append((start + W, i + W))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segs.append((start + W, len(mask) + W))
    return segs

left_segs = find_segments(curvatures > 0.05)
right_segs = find_segments(curvatures < -0.05)
print(f"\n左弯区间: {len(left_segs)} 个")
for s, e in left_segs[:8]:
    print(f"  [{s}-{e}] len={e-s}, avg_curv={curvatures[s-W:e-W].mean():.3f}")
print(f"右弯区间: {len(right_segs)} 个")
for s, e in right_segs[:8]:
    print(f"  [{s}-{e}] len={e-s}, avg_curv={curvatures[s-W:e-W].mean():.3f}")

# 总路径长度
dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
total_len = dists.sum()
print(f"\n总路径长度: {total_len:.1f}m")
