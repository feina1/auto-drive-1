"""Baseline 冻结校验: 检查配置/数据不变量 + 生成归档 sha256 清单.

两个基线 (定义与政策见 BASELINE.md):
- baseline-v1: 图像 5 Hz (18,185 张), 已归档到 output/archive/baseline-v1-hz5/, 只读;
- baseline-v2: 图像 2 Hz (当前有效), 位于 output/ 根目录。
两者除图像采集频率外完全相同: 同专家、同圈速、同场景、同标签规格。

标签规格: 未来 3 s @ 2 Hz 共 6 点, ego 系 (x 前 y 左)。
注意: 标签的 2 Hz 与图像的 2 Hz 是两件事 (一个是预测分辨率, 一个是输入采样率)。

用法 (在 waypoint_pipeline 目录):
    python3 baseline_check.py                    # 校验当前基线 v2
    python3 baseline_check.py --baseline v1      # 校验归档的 v1 (只查数据, 不查 config)
    python3 baseline_check.py --write-manifest   # 校验并写清单
    python3 baseline_check.py --allow-drift      # 重录数据的宽松校验 (±2% 计数, ±0.15m wp0)

退出码: 0 = 全部通过; 1 = 有不通过项。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402

TRAJ = config.OUT_DIR / "trajectory_5laps.json"
NPZ = config.MULTI_LAP_DATASET.with_suffix(".npz")
META = config.MULTI_LAP_DATASET.with_name(config.MULTI_LAP_DATASET.name + "_meta.json")
IMAGES = config.OUT_DIR / "images_trajectory_5laps"
MANIFEST = config.OUT_DIR / "baseline_manifest.json"

# ---- 基线选择: v2 = 当前有效 (output/), v1 = 归档只读 (output/archive/) ----
BASELINES = {
    "v2": {
        "id": "baseline-v2",
        "base": config.OUT_DIR,
        "image_interval_s": 0.5,
        "num_images": 7274,
        "image_bytes": (0.15e9, 0.45e9),
        "check_config": True,    # 当前基线: config 必须是 v2 规格
    },
    "v1": {
        "id": "baseline-v1",
        "base": config.OUT_DIR / "archive" / "baseline-v1-hz5",
        "image_interval_s": 0.2,
        "num_images": 18185,
        "image_bytes": (600e6, 800e6),
        "check_config": False,  # 归档: config 已前进到 v2, 只校验数据不变量与哈希
    },
}

# ---- 冻结的配置 (改动即视为新 baseline, 本检查应失败) ----
_BASE_CONFIG = {
    "BASELINE_ID": "baseline-v2",
    "WP_HORIZON_S": 3.0,
    "WP_FREQ_HZ": 2.0,
    "WP_COUNT": 6,
    "SIM_DT": 0.1,
    "RECORD_HZ": 10,
    "TAIL_EXTRA_S": 6.0,
    "N_LAPS": 5,
    "LAP_SPEEDS_KMH": (40.0, 48.0, 56.0, 64.0, 72.0),
    "IMAGE_INTERVAL_S": 0.5,
    "IMAGE_WIDTH": 640,
    "IMAGE_HEIGHT": 360,
    "IMAGE_FOV_DEG": 60.0,
    "IMAGE_FORMAT": "jpg",
    "IMAGE_QUALITY": 90,
    "TARGET_SPEED_KMH": 70.0,
}

# ---- 归档产物不变量 (严格模式下逐项相等) ----
_BASE_EXPECTED = {
    "num_frames": 36371,
    "num_images": 7274,
    "lap_frames": [9713, 8124, 6963, 6094, 5477],       # 每圈帧数
    "lap_times_s": [971.3, 812.4, 696.3, 609.4, 541.6],
    "lap_speeds_kmh": [40.0, 48.0, 56.0, 64.0, 72.0],
    "wp0_median_m": [5.556, 6.667, 7.778, 8.889, 10.0],  # = 0.5s x 车速
    "offroad_frames": 447,
    "collision_frames": 0,
    "spacing_rel_err_max": 0.01,
    "wp0_max_extra_m": 0.30,                              # wp0_max - median 上限
}


class Checker:
    def __init__(self):
        self.rows = []
        self.fails = []
        self.warns = []

    def ok(self, name, cond, detail=""):
        ok = bool(cond)
        self.rows.append((name, "PASS" if ok else "FAIL", detail))
        if not ok:
            self.fails.append(name)
        return ok

    def warn(self, name, cond, detail=""):
        """不阻断的检查: 失败仅提示 (用于重跑后必然变化的归档哈希)."""
        ok = bool(cond)
        self.rows.append((name, "PASS" if ok else "WARN", detail))
        if not ok:
            self.warns.append(name)
        return ok

    def close(self, name, got, want, tol, unit=""):
        got, want = float(got), float(want)
        return self.ok(name, abs(got - want) <= tol,
                       f"got={got:.4f}{unit} want={want:.4f}{unit} tol={tol}{unit}")

    def report(self):
        width = max(len(r[0]) for r in self.rows) + 2
        for name, status, detail in self.rows:
            print(f"  {status}  {name.ljust(width)} {detail}")
        print(f"\n[baseline] {len(self.rows) - len(self.fails) - len(self.warns)}"
              f"/{len(self.rows)} 项通过")
        if self.fails:
            print("[baseline] 失败项: " + ", ".join(self.fails))
        if self.warns:
            print("[baseline] 警告项: " + ", ".join(self.warns))
        return not self.fails


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_digest(path: Path):
    """目录摘要: 排序后的 (文件名, 字节数) 的 sha256, 返回 (digest, count, bytes)."""
    h = hashlib.sha256()
    count = total = 0
    for p in sorted(path.iterdir()):
        if not p.is_file():
            continue
        size = p.stat().st_size
        h.update(f"{p.name}\t{size}\n".encode("utf-8"))
        count += 1
        total += size
    return h.hexdigest(), count, total


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HERE.parent,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def metadrive_version() -> str:
    for p in (HERE.parent / "metadrive-main" / "metadrive" / "version.py",):
        if p.exists():
            m = re.search(r'VERSION\s*=\s*"([^"]+)"', p.read_text(encoding="utf-8"))
            if m:
                return m.group(1)
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=sorted(BASELINES), default="v2",
                        help="v2=当前有效数据 (默认), v1=已归档的 5 Hz 旧数据 (只读)")
    parser.add_argument("--write-manifest", action="store_true",
                        help="写基线目录下的 baseline_manifest.json (sha256 清单)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="清单路径 (默认 <基线目录>/baseline_manifest.json)")
    parser.add_argument("--allow-drift", action="store_true",
                        help="重录数据宽松校验 (计数 ±2%%, wp0 ±0.15m)")
    parser.add_argument("--skip-hash", action="store_true", help="跳过 sha256 计算")
    parser.add_argument("--strict-hash", action="store_true",
                        help="sha256 不一致计为失败 (默认仅告警: 重跑建集会改 zip 时间戳)")
    args = parser.parse_args()

    # 按基线切换目录与期望值 (局部变量覆盖模块默认, 后续引用均用本次基线)
    spec = BASELINES[args.baseline]
    base = spec["base"]
    TRAJ = base / "trajectory_5laps.json"
    NPZ = base / "dataset_waypoints_5laps.npz"
    META = base / "dataset_waypoints_5laps_meta.json"
    IMAGES = base / "images_trajectory_5laps"
    MANIFEST = Path(args.manifest) if args.manifest else base / "baseline_manifest.json"
    EXPECTED_CONFIG = dict(_BASE_CONFIG, BASELINE_ID=spec["id"],
                           IMAGE_INTERVAL_S=spec["image_interval_s"])
    EXPECTED = dict(_BASE_EXPECTED, num_images=spec["num_images"])

    c = Checker()
    print(f"[baseline] 校验 {spec['id']}  ({base})\n")

    # ---- 1. 配置冻结 (归档基线跳过: config 已前进到新版本) ----
    if spec["check_config"]:
        for key, want in EXPECTED_CONFIG.items():
            got = getattr(config, key)
            if isinstance(want, tuple):
                got = tuple(float(x) for x in got)
            c.ok(f"config.{key}", got == want, f"got={got} want={want}")
    else:
        print("  SKIP  config.*  (归档基线只校验数据, 不校验当前 config)")
        print()

    # 训练侧归一化常数必须与数据一致 (speed/72, steering/0.6)
    try:
        sys.path.insert(0, str(config.TRAIN_DIR))
        import common as train_common
        c.ok("common.SPEED_SCALE_KMH", float(train_common.SPEED_SCALE_KMH) == 72.0,
             f"got={train_common.SPEED_SCALE_KMH}")
        c.ok("common.STEER_SCALE", float(train_common.STEER_SCALE) == 0.6,
             f"got={train_common.STEER_SCALE}")
    except Exception as exc:  # pragma: no cover - 环境异常时给出明确失败
        c.ok("training.common 可导入", False, repr(exc))

    # ---- 2. 文件存在性 ----
    for p in (TRAJ, NPZ, META, IMAGES):
        c.ok(f"存在 {p.name}", p.exists(), str(p))

    if not (TRAJ.exists() and NPZ.exists() and META.exists() and IMAGES.exists()):
        c.report()
        return 1

    traj = json.loads(TRAJ.read_text(encoding="utf-8"))
    meta = json.loads(META.read_text(encoding="utf-8"))
    tm = traj.get("metadata", {})
    drift = bool(args.allow_drift)
    tol_n = 0.02 if drift else 0.0   # 计数相对容差 (drift 模式 ±2%)

    # ---- 3. 轨迹 metadata: 必须是"正常专家跑圈" ----
    c.ok("轨迹.laps", int(tm.get("laps", 0)) == EXPECTED_CONFIG["N_LAPS"], f"got={tm.get('laps')}")
    c.ok("轨迹.planned_laps", int(tm.get("planned_laps", 0)) == EXPECTED_CONFIG["N_LAPS"])
    c.ok("轨迹.lap_speeds", [float(x) for x in tm.get("lap_speeds_kmh", [])]
         == EXPECTED["lap_speeds_kmh"], f"got={tm.get('lap_speeds_kmh')}")
    c.ok("轨迹.record_hz", float(tm.get("record_hz", 0)) == 10.0)
    c.ok("轨迹.aborted", tm.get("aborted") is None, f"got={tm.get('aborted')}")
    c.ok("轨迹.collision_frames", int(tm.get("collision_frames", -1)) == 0,
         f"got={tm.get('collision_frames')}")
    n_frames = int(tm.get("num_frames", 0))
    if drift:
        c.close("轨迹.num_frames", n_frames, EXPECTED["num_frames"], tol_n * EXPECTED["num_frames"], " 帧")
    else:
        c.ok("轨迹.num_frames", n_frames == EXPECTED["num_frames"], f"got={n_frames}")
    c.ok("轨迹.duration≈帧数/10", abs(float(tm.get("duration_s", 0)) - n_frames / 10.0) <= 0.2,
         f"duration={tm.get('duration_s')}")
    offroad = int(tm.get("offroad_frames", 0))
    c.ok("轨迹.offroad_ratio<=5%", offroad / max(n_frames, 1) <= 0.05,
         f"{offroad}/{n_frames}={offroad / max(n_frames, 1):.3%}")
    lap_frames = [int(x) for x in tm.get("lap_frames", [])]
    c.ok("轨迹.lap_frames 长度=5", len(lap_frames) == 5, f"got={lap_frames}")
    c.ok("轨迹.lap_frames 递增", all(b > a for a, b in zip(lap_frames, lap_frames[1:])))
    lap_times = [float(x) for x in tm.get("lap_times_s", [])]
    for i, (got, want) in enumerate(zip(lap_times, EXPECTED["lap_times_s"]), 1):
        if drift:
            c.close(f"轨迹.第{i}圈时长", got, want, 0.02 * want, " s")
        else:
            c.ok(f"轨迹.第{i}圈时长", got == want, f"got={got} want={want}")

    # ---- 4. 数据集 meta ----
    c.ok("meta.baseline_id", meta.get("baseline_id") == EXPECTED_CONFIG["BASELINE_ID"],
         f"got={meta.get('baseline_id')}")
    c.ok("meta.multi_lap", bool(meta.get("multi_lap")), f"got={meta.get('multi_lap')}")
    c.ok("meta.wp_count", int(meta.get("wp_count", 0)) == 6)
    c.close("meta.wp_horizon_s", meta.get("wp_horizon_s", -1), 3.0, 1e-9, " s")
    c.close("meta.wp_freq_hz", meta.get("wp_freq_hz", -1), 2.0, 1e-9, " Hz")
    c.ok("meta.spacing_median_rel_err<=0.01",
         float(meta.get("spacing_median_rel_err", 1)) <= EXPECTED["spacing_rel_err_max"],
         f"got={meta.get('spacing_median_rel_err')}")
    cam = meta.get("camera") or {}
    c.ok("meta.camera", (cam.get("width"), cam.get("height"), cam.get("fov_deg"))
         == (640, 360, 60.0), f"got={cam}")
    c.close("meta.image_interval_s", meta.get("image_interval_s", -1),
            spec["image_interval_s"], 1e-9, " s")
    c.ok("meta.image_format", str(meta.get("image_format")) == "jpg")
    c.ok("meta.image_dir", str(meta.get("image_dir")) == IMAGES.name, f"got={meta.get('image_dir')}")
    c.ok("meta.num_images", int(meta.get("num_images", 0)) == EXPECTED["num_images"],
         f"got={meta.get('num_images')}")

    per_lap = meta.get("per_lap") or []
    c.ok("meta.per_lap 长度=5", len(per_lap) == 5, f"got={len(per_lap)}")
    if len(per_lap) == 5:
        for pl, want_speed, want_frames, want_wp0 in zip(
                per_lap, EXPECTED["lap_speeds_kmh"], EXPECTED["lap_frames"], EXPECTED["wp0_median_m"]):
            tag = f"第{pl.get('lap')}圈"
            c.close(f"meta.{tag}.目标速度", pl.get("target_speed_kmh", -1), want_speed, 1e-6, " km/h")
            if drift:
                c.close(f"meta.{tag}.帧数", pl.get("frames", 0), want_frames, 0.02 * want_frames, " 帧")
            else:
                c.ok(f"meta.{tag}.帧数", int(pl.get("frames", 0)) == want_frames,
                     f"got={pl.get('frames')} want={want_frames}")
            c.close(f"meta.{tag}.wp0 中位", pl.get("wp0_median_m", -1), want_wp0,
                    0.15 if drift else 0.02, " m")
            c.ok(f"meta.{tag}.wp0 波动<=0.3m",
                 float(pl.get("wp0_max_m", 0)) - float(pl.get("wp0_median_m", 0))
                 <= EXPECTED["wp0_max_extra_m"] + 1e-9,
                 f"max-median={float(pl.get('wp0_max_m', 0)) - float(pl.get('wp0_median_m', 0)):.3f} m")
        c.ok("meta.逐圈帧数合计=总帧数", sum(int(p.get("frames", 0)) for p in per_lap) == n_frames,
             f"sum={sum(int(p.get('frames', 0)) for p in per_lap)} total={n_frames}")
        c.ok("meta.逐圈图片合计=总图片数",
             sum(int(p.get("images", 0)) for p in per_lap) == EXPECTED["num_images"])

    # ---- 5. npz 内容 ----
    d = np.load(NPZ, allow_pickle=False)
    required = {"wp_ego", "wp_world", "t", "x", "y", "heading", "speed_kmh",
                "steering", "throttle", "brake", "lap", "lap_t", "target_speed_kmh",
                "yaw_rate", "accel_ms2", "image_path", "train_mask", "val_mask"}
    c.ok("npz.字段齐全", required.issubset(set(d.files)), f"missing={sorted(required - set(d.files))}")
    wp = d["wp_ego"]
    n = wp.shape[0]
    c.ok("npz.wp_ego 形状", wp.shape == (EXPECTED["num_frames"], 6, 2) if not drift
         else abs(n - EXPECTED["num_frames"]) <= 0.02 * EXPECTED["num_frames"],
         f"got={wp.shape}")
    c.ok("npz.wp_ego dtype=float32", wp.dtype == np.float32, str(wp.dtype))
    laps = d["lap"].astype(int)
    uniq = sorted(int(x) for x in np.unique(laps))
    c.ok("npz.lap ∈ {1..5}", uniq == [1, 2, 3, 4, 5], f"got={uniq}")
    if len(per_lap) == 5 and uniq == [1, 2, 3, 4, 5]:
        for pl in per_lap:
            lid = int(pl["lap"])
            c.ok(f"npz.第{lid}圈帧数与 meta 一致",
                 int((laps == lid).sum()) == int(pl["frames"]),
                 f"{(laps == lid).sum()} vs {pl['frames']}")
    speed = d["speed_kmh"].astype(np.float64)
    wp0 = np.linalg.norm(wp[:, 0, :].astype(np.float64), axis=1)
    for lid, want in zip(range(1, 6), EXPECTED["wp0_median_m"]):
        m = laps == lid
        c.close(f"npz.第{lid}圈 wp0 中位", float(np.median(wp0[m])), want,
                0.15 if drift else 0.02, " m")
    # 间距校验: |wp[i+1]-wp[i]| ≈ v*0.5s
    seg = np.linalg.norm(np.diff(wp, axis=1), axis=2).astype(np.float64)
    expected = speed[:, None] / 3.6 * config.WP_DT
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(seg - expected) / np.maximum(expected, 0.5)
    valid = np.broadcast_to(expected > 0.5, rel.shape)
    err = float(np.nanmedian(rel[valid])) if valid.any() else 1.0
    c.ok("npz.间距中位相对误差<=0.01", err <= EXPECTED["spacing_rel_err_max"], f"got={err:.5f}")
    img_paths = d["image_path"]
    n_img_npz = int(sum(1 for p in img_paths if str(p)))
    c.ok(f"npz.带图帧数={EXPECTED['num_images']}",
         n_img_npz == EXPECTED["num_images"], f"got={n_img_npz}")
    tr, va = d["train_mask"], d["val_mask"]
    c.ok("npz.训练/验证不重叠", not bool((tr & va).any()))
    c.ok("npz.训练+验证=全量", int(tr.sum()) + int(va.sum()) == n,
         f"{int(tr.sum())}+{int(va.sum())} vs {n}")
    c.ok("npz.val=第5圈", int(va.sum()) == int((laps == 5).sum()))

    # ---- 6. 图片目录 ----
    digest, n_files, n_bytes = dir_digest(IMAGES)
    lo, hi = spec["image_bytes"]
    c.ok(f"图片.文件数={EXPECTED['num_images']}", n_files == EXPECTED["num_images"], f"got={n_files}")
    c.ok(f"图片.总大小在 {lo / 1e6:.0f}–{hi / 1e6:.0f} MB", lo <= n_bytes <= hi,
         f"{n_bytes / 1e6:.1f} MB")

    # ---- 7. 清单 (sha256) ----
    manifest_path = MANIFEST
    files_info = {}
    if not args.skip_hash:
        for p in (TRAJ, NPZ, META):
            files_info[p.name] = {"sha256": file_sha256(p), "bytes": p.stat().st_size}

    if args.write_manifest:
        manifest = {
            "baseline_id": spec["id"],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": git_commit(),
            "invariants": {
                "num_frames": n_frames,
                "num_images": n_files,
                "lap_speeds_kmh": EXPECTED["lap_speeds_kmh"],
                "lap_times_s": lap_times,
                "lap_frames": [int(p["frames"]) for p in per_lap],
                "wp0_median_m": [float(p["wp0_median_m"]) for p in per_lap],
                "offroad_frames": offroad,
                "collision_frames": int(tm.get("collision_frames", -1)),
                "spacing_median_rel_err": float(meta.get("spacing_median_rel_err", -1)),
            },
            "files": files_info,
            "images": {"dir": IMAGES.name, "count": n_files, "bytes": n_bytes,
                       "digest_sha256": digest},
            "environment": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "panda3d": _safe_version("panda3d"),
                "metadrive": metadrive_version(),
            },
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[baseline] 清单 → {manifest_path}")
        for name, info in files_info.items():
            print(f"  {name:<32} {info['sha256']}  {info['bytes'] / 1e6:.2f} MB")
        print(f"  {IMAGES.name + '/' :<32} {digest}  {n_files} files / {n_bytes / 1e6:.1f} MB")
    elif manifest_path.exists() and not args.skip_hash:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = c.ok if args.strict_hash else c.warn
        for name, info in (old.get("files") or {}).items():
            same = files_info.get(name, {}).get("sha256") == info.get("sha256")
            record(f"sha256.{name}", same,
                   "与清单一致" if same else "与清单不一致 (重跑过建集? 用 --strict-hash 可计为失败)")
        if old.get("images", {}).get("digest_sha256"):
            same = old["images"]["digest_sha256"] == digest
            record("sha256.图片目录", same, "与清单一致" if same else "与清单不一致 (图片目录被改动过)")
    else:
        print("\n[baseline] 未找到清单, 跳过 sha256 比对 (可用 --write-manifest 生成)")

    print()
    ok = c.report()
    return 0 if ok else 1


def _safe_version(pkg: str) -> str:
    try:
        mod = __import__(pkg)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())