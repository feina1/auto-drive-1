"""离线单图诊断: 把训练 / 离线预测 / 闭环测试压成一张 PNG, 供人工快速判读.

用途: 训练机上没有方便的图形界面时, 只拷一张 summary.png 就能看出问题:
- 训练曲线: 是否收敛/过拟合/早停;
- 分点 RMSE: 近端 vs 远端退化;
- 预测样本: 系统性偏短/偏长、切弯;
- ADE × 车速 / |yaw_rate|: 速度泛化与弯道失败;
- 闭环地图与时序: 是否跑偏、出界、撞车、跑不起来;
- 自动诊断: 把可判定的问题直接写成文字.

版本化: 每次代码/策略调整后重跑并指定新版本, 图与指标存到 reports/<v>/,
同时更新 reports/latest/, 人工点评记录在 ITERATION_LOG.md (只内部交流)。

用法 (在 waypoint_pipeline 目录):
    python training/report.py --version v1 \
        --ckpt output/training/smoke_5laps/tinycnn_best.pt \
        --predict-dir output/training/smoke_5laps/predict_full \
        --closed-loop-dir output/training/smoke_5laps/closed_loop

    python training/report.py --version auto --predict-dir auto --closed-loop-dir auto
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

TRAINING_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = TRAINING_DIR.parent
PROJECT_ROOT = PIPELINE_DIR.parent
for _p in (str(PIPELINE_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402

REPORT_ROOT = PIPELINE_DIR / "reports"
PASS_COLOR, WARN_COLOR, FAIL_COLOR, INFO_COLOR = "#1b8a3a", "#c98a00", "#c0392b", "#607d8b"

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "Arial Unicode MS",
                        "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
                        "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------- 读取工具
def load_json(path):
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return None


def load_rows(path):
    if not path or not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def newest_metrics(pattern: str):
    hits = sorted(PIPELINE_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def resolve_predict_dir(arg: str | None):
    if arg and arg != "auto":
        return Path(arg)
    if arg == "auto":
        hit = newest_metrics("output/**/pred_metrics.json")
        return hit.parent if hit else None
    return newest_metrics("output/**/pred_metrics.json").parent


def resolve_closed_dir(arg: str | None):
    if arg and arg != "auto":
        return Path(arg)
    if arg == "auto":
        hit = newest_metrics("output/**/closed_loop_metrics.json")
        return hit.parent if hit else None
    return None


def find_npz(explicit: str | None, pred_metrics, split):
    if explicit:
        return Path(explicit)
    for cand in ((split or {}).get("npz"), (pred_metrics or {}).get("npz")):
        if cand and Path(cand).exists():
            return Path(cand)
    multi = config.MULTI_LAP_DATASET.with_suffix(".npz")
    return multi if multi.exists() else config.OUT_DIR / "dataset_waypoints.npz"


def find_train_dir(ckpt: Path | None, predict_dir: Path | None):
    if ckpt and ckpt.exists():
        return ckpt.parent
    if predict_dir is not None:
        return predict_dir.parent
    return None


def code_fingerprint() -> str:
    files = [PIPELINE_DIR / "config.py", PIPELINE_DIR / "baseline_check.py",
             PIPELINE_DIR / "collect_trajectory.py", PIPELINE_DIR / "make_waypoints.py",
             PIPELINE_DIR / "controllers.py", PIPELINE_DIR / "traj_utils.py"]
    files += sorted((TRAINING_DIR).glob("*.py"))
    h = hashlib.sha256()
    for p in files:
        if p.exists():
            h.update(p.name.encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- 计算工具
def ade_fde(pred, gt):
    d = np.linalg.norm(pred - gt, axis=-1)
    return float(d.mean()), float(d[-1].mean()), d.mean(axis=0)


def grade(v, pass_thr, warn_thr):
    if v is None:
        return "info"
    return "pass" if v <= pass_thr else ("warn" if v <= warn_thr else "fail")


def status_color(status):
    return {"pass": PASS_COLOR, "warn": WARN_COLOR, "fail": FAIL_COLOR}.get(status, INFO_COLOR)


def draw_chip(ax, x, y, w, h, label, value, status):
    color = status_color(status)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                transform=ax.transAxes, facecolor=color + "18",
                                edgecolor=color, linewidth=1.2))
    ax.text(x + w / 2, y + h * 0.66, label, ha="center", va="center", fontsize=8,
            color="#444444", transform=ax.transAxes)
    ax.text(x + w / 2, y + h * 0.30, value, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=color, transform=ax.transAxes)


def wrap_bullets(bullets, width=28):
    out = []
    for b in bullets:
        wrapped = textwrap.wrap(b, width=width, break_long_words=True) or [""]
        out.append("· " + wrapped[0])
        out.extend("  " + w for w in wrapped[1:])
    return out


# ---------------------------------------------------------------- 面板
def panel_training(ax, history, split):
    if not history:
        ax.axis("off")
        ax.text(0.5, 0.5, "无训练历史", ha="center", va="center", color="#999999")
        return
    ep = [r["epoch"] for r in history]
    tr = [r.get("train_loss") for r in history]
    ade = [r.get("ade_m") for r in history]
    ax.plot(ep, tr, color="#37474f", lw=1.4, label="train loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax2 = ax.twinx()
    ax2.plot(ep, ade, color="#e67e22", lw=1.6, label="val ADE")
    ax2.set_ylabel("val ADE (m)", color="#e67e22")
    ax2.tick_params(axis="y", colors="#e67e22")
    if split and split.get("best_val_ade_m") is not None:
        best = int(split.get("best_epoch", len(ep)))
        ax2.axvline(best, color="#e67e22", ls="--", lw=0.8, alpha=0.6)
        ax2.annotate(f"best ep{best}\n{split['best_val_ade_m']:.2f} m",
                     xy=(best, split["best_val_ade_m"]), xytext=(0.55, 0.75),
                     textcoords="axes fraction", fontsize=8, color="#b35300")
    ax2.set_ylim(0, max(10.0, max(a for a in ade if a is not None) * 1.2))
    if len(ep) <= 2:
        ax.text(0.02, 0.12, f"仅 {len(ep)} 个 epoch（冒烟）", transform=ax.transAxes,
                fontsize=8, color="#999999")
    ax.set_title("训练曲线")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7, framealpha=0.7)


def panel_rmse(ax, rmse):
    if rmse is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "无分点误差", ha="center", va="center", color="#999999")
        return
    x = np.arange(1, len(rmse) + 1)
    colors = plt.cm.YlOrRd(np.linspace(0.25, 0.85, len(rmse)))
    bars = ax.bar(x, rmse, color=colors, edgecolor="#8d6e63", linewidth=0.5)
    for b, v in zip(bars, rmse):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [f"{0.5 * i:.1f}s" for i in x])
    ax.set_ylabel("RMSE (m)")
    ax.set_title("分点 RMSE（近端→远端）")
    ax.margins(y=0.18)


def panel_speed(ax, rows, meta):
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "无逐帧预测", ha="center", va="center", color="#999999")
        return
    sp = np.array([float(r["speed_kmh"]) for r in rows])
    ad = np.array([float(r["ade_m"]) for r in rows])
    ax.scatter(sp, ad, s=8, alpha=0.25, color="#546e7a", edgecolors="none")
    order = np.argsort(sp)
    bins = np.linspace(np.percentile(sp, 1), np.percentile(sp, 99), 7)
    med_x, med_y = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (sp >= lo) & (sp <= hi)
        if m.sum() >= max(5, len(sp) // 100):
            med_x.append(float(np.median(sp[m])))
            med_y.append(float(np.median(ad[m])))
    ax.plot(med_x, med_y, color="#c0392b", lw=1.6, marker="o", ms=3, label="分桶中位")
    for v in (config.LAP_SPEEDS_KMH if hasattr(config, "LAP_SPEEDS_KMH") else ()):
        ax.axvline(v, color="#b0bec5", lw=0.7, ls=":")
    ax.set_xlabel("车速 (km/h)")
    ax.set_ylabel("逐帧 ADE (m)")
    ax.set_title("ADE × 车速")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(0, max(5.0, float(np.percentile(ad, 99)) * 1.15))
    if np.std(sp) < 2.0:
        ax.text(0.98, 0.03, f"验证帧仅单一速度 ≈{np.mean(sp):.0f} km/h",
                transform=ax.transAxes, ha="right", fontsize=7.5, color="#777777")
    ax.margins(x=0.03)


def panel_yaw(ax, rows, ade_raw):
    if not rows or ade_raw is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "无弯道数据", ha="center", va="center", color="#999999")
        return
    idx = np.array([int(r["src_index"]) for r in rows])
    yr = np.array([abs(float(ade_raw["yaw_rate"][i])) if i < len(ade_raw["yaw_rate"]) else np.nan
                   for i in idx])
    ad = np.array([float(r["ade_m"]) for r in rows])
    m = np.isfinite(yr)
    ax.scatter(yr[m], ad[m], s=8, alpha=0.22, color="#6a1b9a", edgecolors="none")
    bins = np.array([0, 0.02, 0.05, 0.1, 0.2, 1.0])
    mx, my = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mm = m & (yr >= lo) & (yr < hi)
        if mm.sum() >= max(5, len(idx) // 100):
            mx.append(float(np.median(yr[mm])))
            my.append(float(np.median(ad[mm])))
    ax.plot(mx, my, color="#6a1b9a", lw=1.6, marker="s", ms=3)
    ax.set_xlabel("|yaw_rate| (rad/s)")
    ax.set_ylabel("逐帧 ADE (m)")
    ax.set_title("ADE × 弯道强度")
    ax.set_ylim(0, max(5.0, float(np.percentile(ad[m], 99)) * 1.15))
    ax.margins(x=0.03)


def panel_samples(fig, spec, arrays, rows):
    if arrays is None or not rows:
        ax = fig.add_subplot(spec)
        ax.axis("off")
        ax.text(0.5, 0.5, "无预测样本", ha="center", va="center", color="#999999")
        return
    sub = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.32)
    pred, gt = arrays["pred"], arrays["gt"]
    d = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)
    n = len(d)
    picks = [int(np.argmin(d)), int(np.argsort(d)[n // 2]), int(np.argmax(d))]
    labels = ["最好", "中位", "最差"]
    for k, (pos, lab) in enumerate(zip(picks, labels)):
        ax = fig.add_subplot(sub[0, k])
        ax.plot(gt[pos, :, 0], gt[pos, :, 1], "-o", color="#2c3e50", ms=3, lw=1.6, label="GT")
        ax.plot(pred[pos, :, 0], pred[pos, :, 1], "--o", color="#e74c3c", ms=3, lw=1.6, label="pred")
        ax.scatter([0], [0], marker="*", s=90, color="#1b8a3a", zorder=5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{lab} #{int(arrays['src_index'][pos])} ADE={d[pos]:.1f}m", fontsize=9)
        ax.set_xlabel("x 前向 (m)", fontsize=7)
        if k == 0:
            ax.set_ylabel("y 左向 (m)", fontsize=7)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.text(0.02, 0.03,
                    f"pred 均长 {np.linalg.norm(pred, axis=-1).mean():.1f} m / "
                    f"GT {np.linalg.norm(gt, axis=-1).mean():.1f} m",
                    transform=ax.transAxes, fontsize=7, color="#777777", va="bottom")
        if k == 2 and np.linalg.norm(pred, axis=-1).mean() < 2.0:
            ax.annotate("pred 堆在原点", xy=(0, 0), xytext=(0.30, 0.14),
                        textcoords="axes fraction", fontsize=8, color="#e74c3c",
                        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.0))
        if k == 2:
            ax.legend(fontsize=7, loc="upper left")


def panel_diagnosis(ax, bullets, split, closed):
    ax.axis("off")
    ax.set_title("自动诊断（详细点评见 ITERATION_LOG.md）", loc="left")
    lines = wrap_bullets(bullets, width=44)
    y = 0.95
    step = min(0.055, 0.88 / max(len(lines), 1))
    for line in lines[:24]:
        ax.text(0.0, y, line, transform=ax.transAxes, fontsize=8.4, va="top", color="#333333")
        y -= step
    if split:
        note = (f"训练 {split.get('train_samples')} / 验证 {split.get('val_samples')} 帧"
                f" · val_laps={split.get('val_laps')}")
        ax.text(0.0, 0.0, note, transform=ax.transAxes, fontsize=7.5,
                color="#888888", va="bottom")


def panel_closed_map(ax, closed_dir, log_rows):
    ref = load_json(PROJECT_ROOT / "simple_track_reference_clean.json")
    if ref:
        rp = ref.get("reference_path", {})
        ax.plot(rp.get("x", []), rp.get("y", []), color="#b0bec5", lw=1.0, label="参考线")
    if not log_rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "无闭环数据", ha="center", va="center", color="#999999")
        return
    x = np.array([float(r["x"]) for r in log_rows])
    y = np.array([float(r["y"]) for r in log_rows])
    lat = np.array([abs(float(r["lat_err_m"])) for r in log_rows])
    off = np.array([str(r.get("offroad", "")).lower() == "true" for r in log_rows])
    sc = ax.scatter(x, y, c=lat, s=9, cmap="viridis", vmin=0, vmax=max(2.5, float(np.percentile(lat, 95))))
    cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("|横向误差| (m)", fontsize=7)
    ax.scatter(x[off], y[off], marker="x", s=22, color="#e74c3c", label="出界帧", zorder=4)
    ax.scatter([x[0]], [y[0]], marker="o", s=55, color="#1b8a3a", label="起点", zorder=5)
    ax.scatter([x[-1]], [y[-1]], marker="s", s=38, color="#111111", label="终点", zorder=5)
    span = float(max(x.max() - x.min(), y.max() - y.min()))
    if span < 200.0:
        pad = max(20.0, span * 0.5 + 10.0)
        cx, cy = float((x.min() + x.max()) / 2), float((y.min() + y.max()) / 2)
        ax.set_xlim(cx - pad, cx + pad)
        ax.set_ylim(cy - pad, cy + pad)
        ax.text(0.02, 0.04, f"自动放大：轨迹跨度 {span:.1f} m",
                transform=ax.transAxes, fontsize=7.5, color="#666666")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("闭环轨迹（颜色=横向误差）")
    ax.set_xlabel("x (m)", fontsize=7)
    ax.set_ylabel("y (m)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="best")


def panel_closed_time(ax, log_rows, metrics, target_speed):
    if not log_rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "无闭环数据", ha="center", va="center", color="#999999")
        return
    t = np.array([float(r["t"]) for r in log_rows])
    sp = np.array([float(r["speed_kmh"]) for r in log_rows])
    lat = np.array([abs(float(r["lat_err_m"])) for r in log_rows])
    off = np.array([str(r.get("offroad", "")).lower() == "true" for r in log_rows])
    ax.plot(t, sp, color="#1565c0", lw=1.5, label="车速")
    if target_speed:
        ax.axhline(target_speed, color="#1565c0", ls="--", lw=0.8, alpha=0.6)
    ax.set_ylabel("车速 (km/h)", color="#1565c0")
    ax.tick_params(axis="y", colors="#1565c0")
    ax2 = ax.twinx()
    ax2.plot(t, lat, color="#c0392b", lw=1.3, label="|横向误差|")
    ax2.set_ylabel("|横向误差| (m)", color="#c0392b")
    ax2.tick_params(axis="y", colors="#c0392b")
    if off.any():
        y0, y1 = ax.get_ylim()
        ax.fill_between(t, y0, y1, where=off, color="#e74c3c", alpha=0.12, step="mid")
    ax.set_xlabel("时间 (s)")
    ax.set_title("闭环时序（阴影=出界）")
    ax.text(0.99, 0.02, "出界帧占比 %.2f%%" % (100 * off.mean()),
            transform=ax.transAxes, ha="right", fontsize=7.5, color="#777777")


# ---------------------------------------------------------------- 诊断文本
def build_diagnosis(metrics, rows, closed, split, meta, arrays=None):
    b = []
    ade = (metrics or {}).get("ade_m")
    fde = (metrics or {}).get("fde_m")
    if arrays is not None:
        pn = float(np.linalg.norm(arrays["pred"], axis=-1).mean())
        gn = float(np.linalg.norm(arrays["gt"], axis=-1).mean())
        if gn > 5.0 and pn < max(1.0, 0.1 * gn):
            b.append(f"预测坍缩到原点（均长 {pn:.1f} m vs GT {gn:.1f} m）："
                     f"head 未学会输出尺度，ADE≈GT 平均长度。")
    if ade is not None:
        if ade > 5:
            b.append(f"离线 ADE {ade:.1f} m、FDE {fde:.1f} m：尚未学会，仅用于验证链路。")
        elif ade > 3:
            b.append(f"离线 ADE {ade:.1f} m：未达 3 m 目标，需要继续训练/查数据。")
        else:
            b.append(f"离线 ADE {ade:.1f} m、FDE {fde:.1f} m：达到离线可用线。")
        if fde and ade and fde / ade > 2.2:
            b.append(f"FDE/ADE={fde / ade:.1f}>2.2：远端退化明显，优先查时域/数据而非 head。")
    rmse = (metrics or {}).get("rmse_per_point_m")
    if rmse and len(rmse) >= 2 and rmse[-1] / max(rmse[0], 1e-6) > 8:
        b.append("首点误差已很大，远端必然发散：先看样本图是否系统性偏短/偏长。")
    if split and (split.get("train_samples") or 0) < 2000:
        b.append(f"训练样本仅 {split.get('train_samples')} 帧：结论不代表模型能力，先扩到全量。")
    if meta:
        per = meta.get("per_lap") or []
        train_laps = [p for p in per if int(p["lap"]) != int((split or {}).get("val_laps", [5])[0])]
        val_speed = next((p["target_speed_kmh"] for p in per
                          if int(p["lap"]) == int((split or {}).get("val_laps", [5])[0])), None)
        train_speeds = {p["target_speed_kmh"] for p in train_laps}
        if val_speed is not None and val_speed not in train_speeds:
            b.append(f"验证圈 {val_speed:.0f} km/h 是训练未见速度：重点看速度分桶与闭环高速表现。")
    if closed:
        if closed.get("aborted") == "max_seconds":
            b.append("闭环被时长截断（冒烟），不是完圈结果，不能据此判通过。")
        elif not closed.get("completed"):
            b.append("闭环未完圈：先看地图与时序，判断是跑偏、卡死还是撞车。")
        if closed.get("collision_frames"):
            b.append(f"闭环碰撞 {closed['collision_frames']} 帧：优先级最高，先排除安全风险。")
        if closed.get("offroad_ratio", 0) > config.EVAL_OFFROAD_MAX_RATIO:
            b.append(f"出界率 {closed['offroad_ratio']:.2%} 超阈值 "
                     f"{config.EVAL_OFFROAD_MAX_RATIO:.0%}：看地图上红叉分布。")
        tgt = closed.get("target_speed_kmh") or 0
        if tgt and closed.get("speed_mean_kmh", 0) < 0.6 * tgt:
            b.append(f"闭环均速仅 {closed['speed_mean_kmh']:.1f}/{tgt:.0f} km/h：预测过短或控制器几乎不前进。")
        if closed.get("infer_ms", 0) > 100:
            b.append(f"单次推理 {closed['infer_ms']:.0f} ms：10 Hz 实时性不足，需 GPU 或 --infer-every。")
    ctrl = load_json(config.OUT_DIR / "validate_controller" / "metrics.json")
    if ctrl and ctrl.get("passed"):
        b.append(f"GT 控制器基线通过（均速 {ctrl.get('speed_mean_kmh', 0):.0f} km/h，"
                 f"平均横向 {ctrl.get('lat_abs_mean_m', 0):.3f} m）：问题不在控制器。")
    if not b:
        b.append("数据不足，无法自动诊断。")
    return b


# ---------------------------------------------------------------- 主流程
def next_version(root: Path) -> str:
    nums = [int(p.name[1:]) for p in root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
    return f"v{max(nums, default=0) + 1}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--predict-dir", type=str, default="auto")
    ap.add_argument("--closed-loop-dir", type=str, default=None)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--version", type=str, default="auto")
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--note", type=str, default="")
    ap.add_argument("--out-root", type=str, default=str(REPORT_ROOT))
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else None
    predict_dir = resolve_predict_dir(args.predict_dir)
    closed_dir = resolve_closed_dir(args.closed_loop_dir)
    if predict_dir is None or not (predict_dir / "pred_metrics.json").exists():
        raise SystemExit("未找到 pred_metrics.json，先跑 training/predict.py（或传 --predict-dir）")

    metrics = load_json(predict_dir / "pred_metrics.json")
    rows = load_rows(predict_dir / "pred_per_frame.csv")
    arrays_path = predict_dir / "pred_arrays.npz"
    arrays = dict(np.load(arrays_path, allow_pickle=False)) if arrays_path.exists() else None

    train_dir = find_train_dir(ckpt, predict_dir)
    backbone = (metrics or {}).get("backbone", "unknown")
    history = load_json(train_dir / f"{backbone}_history.json") if train_dir else None
    split = load_json(train_dir / f"{backbone}_split.json") if train_dir else None

    npz_path = find_npz(args.dataset, metrics, split)
    meta = load_json(npz_path.with_name(npz_path.stem + "_meta.json"))
    ade_raw = None
    if npz_path.exists() and rows:
        ade_raw = dict(np.load(npz_path, allow_pickle=False))

    closed = load_json(closed_dir / "closed_loop_metrics.json") if closed_dir else None
    closed_rows = load_rows(closed_dir / "closed_loop_log.csv") if closed_dir else []
    if closed is None:
        closed_dir = None

    # ---- 诊断 ----
    bullets = build_diagnosis(metrics, rows, closed, split, meta, arrays=arrays)

    # ---- 画布 ----
    fig = plt.figure(figsize=(19, 14.5))
    gs = gridspec.GridSpec(4, 5, figure=fig, height_ratios=[0.52, 1.0, 1.0, 1.05],
                           hspace=0.42, wspace=0.32, left=0.045, right=0.975,
                           top=0.965, bottom=0.05)

    # 头部
    axh = fig.add_subplot(gs[0, :])
    axh.axis("off")
    title = args.title or f"Waypoint 诊断 {args.version}"
    axh.text(0.0, 0.94, title, fontsize=17, fontweight="bold", va="top")
    sub = (f"{backbone} · val_laps={ (split or {}).get('val_laps') } · "
           f"最佳 epoch={ (split or {}).get('best_epoch') } · "
           f"数据 {(meta or {}).get('baseline_id', '?')} · "
           f"生成 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    axh.text(0.0, 0.62, sub, fontsize=9.5, color="#555555", va="top")

    ade, fde = (metrics or {}).get("ade_m"), (metrics or {}).get("fde_m")
    closed_status = "info"
    if closed:
        if closed.get("aborted") == "max_seconds":
            closed_status = "info"
        elif not closed.get("completed") or closed.get("collision_frames"):
            closed_status = "fail"
        elif closed.get("offroad_ratio", 0) > config.EVAL_OFFROAD_MAX_RATIO:
            closed_status = "warn"
        else:
            closed_status = "pass"
    chips = [
        ("离线 ADE", f"{ade:.1f} m" if ade is not None else "—", grade(ade, 3.0, 5.0)),
        ("离线 FDE", f"{fde:.1f} m" if fde is not None else "—", grade(fde, 5.0, 8.0)),
        ("训练/验证帧", f"{(split or {}).get('train_samples', '—')}/"
                        f"{(split or {}).get('val_samples', '—')}", "info"),
        ("闭环状态", ("截断" if (closed or {}).get("aborted") == "max_seconds"
                      else ("完圈" if (closed or {}).get("completed") else ("无数据" if not closed else "未完圈"))),
         closed_status),
        ("碰撞/出界", (f"{closed.get('collision_frames', 0)}/{closed.get('offroad_ratio', 0):.1%}"
                       if closed else "—"),
         "info" if not closed else ("fail" if closed.get("collision_frames") else
                                    ("warn" if closed.get("offroad_ratio", 0) > config.EVAL_OFFROAD_MAX_RATIO else "pass"))),
        ("均速/目标", (f"{closed.get('speed_mean_kmh', 0):.0f}/{closed.get('target_speed_kmh', 0):.0f}"
                       if closed else "—"), "info"),
        ("推理耗时", f"{closed['infer_ms']:.0f} ms" if closed else "—",
         "info" if not closed else ("pass" if closed.get("infer_ms", 0) <= 100 else "warn")),
    ]
    x0, w, gap = 0.0, 0.132, 0.008
    for i, (lab, val, st) in enumerate(chips):
        draw_chip(axh, x0 + i * (w + gap), 0.0, w, 0.42, lab, val, st)

    # 面板
    panel_training(fig.add_subplot(gs[1, 0:3]), history, split)
    panel_rmse(fig.add_subplot(gs[1, 3]), (metrics or {}).get("rmse_per_point_m"))
    panel_speed(fig.add_subplot(gs[1, 4]), rows, meta)
    panel_samples(fig, gs[2, 0:2], arrays, rows)
    panel_yaw(fig.add_subplot(gs[2, 2]), rows, ade_raw)
    panel_diagnosis(fig.add_subplot(gs[2, 3:5]), bullets, split, closed)
    panel_closed_map(fig.add_subplot(gs[3, 0:3]), closed_dir, closed_rows)
    panel_closed_time(fig.add_subplot(gs[3, 3:5]), closed_rows, closed,
                      (closed or {}).get("target_speed_kmh"))

    fp = code_fingerprint()
    fig.text(0.045, 0.012,
             f"仅内部迭代用 · 数据固定 { (meta or {}).get('baseline_id', 'baseline-v1') }（不做任何数据改动） · "
             f"code_fingerprint={fp} · 最新版: reports/latest/summary.png",
             fontsize=8, color="#888888")

    # ---- 落盘 ----
    root = Path(args.out_root)
    version = next_version(root) if args.version == "auto" else args.version
    out_dir = root / version
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "summary.png"
    fig.savefig(png, dpi=115)
    plt.close(fig)

    snapshot = {
        "version": version,
        "title": args.title,
        "note": args.note,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_id": (meta or {}).get("baseline_id"),
        "code_fingerprint": fp,
        "sources": {
            "ckpt": str(ckpt) if ckpt else None,
            "predict_dir": str(predict_dir),
            "closed_loop_dir": str(closed_dir) if closed_dir else None,
            "dataset": str(npz_path),
        },
        "metrics": {
            "ade_m": ade, "fde_m": fde,
            "rmse_per_point_m": (metrics or {}).get("rmse_per_point_m"),
            "val_laps": (split or {}).get("val_laps"),
            "train_samples": (split or {}).get("train_samples"),
            "val_samples": (split or {}).get("val_samples"),
            "best_epoch": (split or {}).get("best_epoch"),
            "closed_loop": closed,
        },
        "diagnosis": bullets,
    }
    (out_dir / "summary_metrics.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    latest = root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png, latest / "summary.png")
    shutil.copy2(out_dir / "summary_metrics.json", latest / "summary_metrics.json")

    print(f"[report] {version} → {png}")
    print(f"[report] latest → {latest / 'summary.png'}")
    for b in bullets:
        print(f"[report] · {b}")


if __name__ == "__main__":
    main()