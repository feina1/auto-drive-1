#!/usr/bin/env python3
"""轨迹可视化 — 生成两张对比图：轨迹 + 横向误差曲线"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAJ_DIR = os.path.join(SCRIPT_DIR, 'trajectories')
PARAMS_CSV = os.path.join(SCRIPT_DIR, 'params.csv')
RESULTS_CSV = os.path.join(SCRIPT_DIR, 'results_summary.csv')

# 参考路径（用于计算横向误差）
BASE_DIR = os.path.dirname(SCRIPT_DIR)
REF_PATH = os.path.join(BASE_DIR, 'simple_track_reference_clean.json')


def load_params():
    experiments = []
    with open(PARAMS_CSV, 'r') as f:
        for row in csv.DictReader(f):
            experiments.append(row)
    return experiments


def load_results():
    results = {}
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'r') as f:
            for row in csv.DictReader(f):
                results[row['experiment_id']] = row
    return results


def load_ref_path():
    import json
    with open(REF_PATH, 'r') as f:
        data = json.load(f)
    return np.array(data['reference_path']['x']), np.array(data['reference_path']['y'])


def compute_lateral_errors(traj_x, traj_y, ref_x, ref_y):
    """计算每个轨迹点的横向误差（到参考路径的最近距离）"""
    # 降采样参考路径以加速（保持足够精度）
    step = max(1, len(ref_x) // 2000)
    rx, ry = ref_x[::step], ref_y[::step]
    # 向量化计算：每个轨迹点到所有参考点的最小距离
    # 分批处理避免内存爆炸
    batch = 500
    errors = np.zeros(len(traj_x))
    for i in range(0, len(traj_x), batch):
        bx = traj_x[i:i+batch]
        by = traj_y[i:i+batch]
        # (batch, n_ref) distance matrix
        dx = bx[:, None] - rx[None, :]
        dy = by[:, None] - ry[None, :]
        dists = np.sqrt(dx**2 + dy**2)
        errors[i:i+batch] = dists.min(axis=1)
    return errors


def plot_trajectories(experiments, results, ref_x, ref_y):
    """图1: 轨迹对比"""
    n = len(experiments)
    cols, rows = 4, (n + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, exp in enumerate(experiments):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        exp_id = exp['experiment_id']
        traj_file = os.path.join(TRAJ_DIR, f"{exp_id}.npz")

        if os.path.exists(traj_file):
            data = np.load(traj_file)
            traj_x, traj_y = data['traj_x'], data['traj_y']
            step = max(1, len(ref_x) // 2000)
            ax.plot(ref_x[::step], ref_y[::step], 'b-', linewidth=0.5, alpha=0.4, label='Ref')
            ax.plot(traj_x, traj_y, 'r-', linewidth=0.6, alpha=0.8, label='Traj')
            ax.plot(traj_x[0], traj_y[0], 'go', markersize=5, label='Start')
            ax.plot(traj_x[-1], traj_y[-1], 'rs', markersize=5, label='End')
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')

        res = results.get(exp_id, {})
        title = (f"{exp_id}\n"
                 f"kp={exp['kp_steer']} ki={exp['ki_steer']} kd={exp['kd_steer']}\n"
                 f"look={exp['look_dist_base']} scale={exp['look_dist_scale']} "
                 f"max_s={exp['max_steer']}\n"
                 f"Laps={res.get('laps','?')} MaxErr={res.get('max_lat_err','?')}m "
                 f"AvgErr={res.get('avg_lat_err','?')}m")
        ax.set_title(title, fontsize=7, fontfamily='monospace')
        ax.set_aspect('equal')
        ax.tick_params(labelsize=5)
        ax.legend(loc='upper right', fontsize=5)
        ax.grid(True, linewidth=0.2, alpha=0.3)

    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis('off')

    fig.suptitle('PID Parameter Comparison — Vehicle Trajectories',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'param_comparison_trajectories.jpg')
    fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ 轨迹对比图: {out}")


def plot_lateral_errors(experiments):
    """图2: 横向误差随时间变化（使用运行时真实数据）"""
    n = len(experiments)
    cols, rows = 4, (n + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, exp in enumerate(experiments):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        exp_id = exp['experiment_id']
        traj_file = os.path.join(TRAJ_DIR, f"{exp_id}.npz")

        if os.path.exists(traj_file):
            data = np.load(traj_file)
            # 优先使用运行时保存的真实横向误差
            if 'lat_errors' in data:
                errors = data['lat_errors']
            else:
                errors = compute_lateral_errors(data['traj_x'], data['traj_y'], ref_x, ref_y)
            frames = np.arange(len(errors))
            ax.plot(frames, errors, 'r-', linewidth=0.8, alpha=0.8)
            ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=0.8, alpha=0.6, label='1m threshold')
            ax.axhline(y=np.mean(errors), color='blue', linestyle=':', linewidth=0.8, alpha=0.6,
                       label=f'avg={np.mean(errors):.2f}m')
            ax.set_xlabel('Frame', fontsize=7)
            ax.set_ylabel('Lateral Error (m)', fontsize=7)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')

        title = (f"{exp_id}\n"
                 f"kp={exp['kp_steer']} ki={exp['ki_steer']} kd={exp['kd_steer']} "
                 f"look={exp['look_dist_base']}")
        ax.set_title(title, fontsize=7, fontfamily='monospace')
        ax.tick_params(labelsize=6)
        ax.legend(loc='upper right', fontsize=5)
        ax.grid(True, linewidth=0.2, alpha=0.3)
        ax.set_ylim(bottom=0)

    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis('off')

    fig.suptitle('PID Parameter Comparison — Lateral Error Over Time',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'param_comparison_errors.jpg')
    fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ 误差对比图: {out}")


def plot_summary_bar(experiments, results):
    """图3: 柱状图汇总对比"""
    exp_ids = [e['experiment_id'] for e in experiments]
    max_errs = [float(results.get(eid, {}).get('max_lat_err', 0)) for eid in exp_ids]
    avg_errs = [float(results.get(eid, {}).get('avg_lat_err', 0)) for eid in exp_ids]

    x = np.arange(len(exp_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))
    bars1 = ax.bar(x - width/2, max_errs, width, label='Max Lat Error (m)', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, avg_errs, width, label='Avg Lat Error (m)', color='blue', alpha=0.7)

    ax.set_xlabel('Experiment', fontsize=10)
    ax.set_ylabel('Error (m)', fontsize=10)
    ax.set_title('PID Parameter Comparison — Error Summary', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # 标注数值
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=6)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'param_comparison_summary.jpg')
    fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ 汇总柱状图: {out}")


def main():
    experiments = load_params()
    results = load_results()
    ref_x, ref_y = load_ref_path()

    print(f"共 {len(experiments)} 个实验")
    plot_trajectories(experiments, results, ref_x, ref_y)
    plot_lateral_errors(experiments)
    plot_summary_bar(experiments, results)
    print("✓ 全部完成")


if __name__ == "__main__":
    main()
