#!/usr/bin/env python3
"""参数实验运行器 — 无头模式跑所有参数组合，保存轨迹数据"""
import sys
import os
import csv
import json
import numpy as np

# 导入上级目录的模块（不修改原始文件）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_track import LANE_NUM, LANE_WIDTH, TARGET_SPEED_KMH, LapCounter, SimpleTrackEnv
from pid_demo_autonomous import PIDController, PathFollower

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_CSV = os.path.join(SCRIPT_DIR, 'params.csv')
TRAJ_DIR = os.path.join(SCRIPT_DIR, 'trajectories')
os.makedirs(TRAJ_DIR, exist_ok=True)

# 每个实验跑的帧数
FRAMES_PER_EXP = 6000


def load_params():
    """从 CSV 加载参数配置"""
    experiments = []
    with open(PARAMS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            experiments.append({
                'id': row['experiment_id'],
                'kp_steer': float(row['kp_steer']),
                'ki_steer': float(row['ki_steer']),
                'kd_steer': float(row['kd_steer']),
                'look_dist_base': float(row['look_dist_base']),
                'look_dist_scale': float(row['look_dist_scale']),
                'max_steer': float(row['max_steer']),
                'description': row['description'],
            })
    return experiments


def make_env():
    """创建无头 MetaDrive 环境"""
    config = dict(
        use_render=False,
        manual_control=False,
        traffic_density=0.0,
        num_scenarios=1,
        start_seed=0,
        map_region_size=4096,
        random_lane_width=False,
        random_lane_num=False,
        out_of_route_done=False,
        on_continuous_line_done=False,
        map_config=dict(lane_num=LANE_NUM, lane_width=LANE_WIDTH),
        traffic_vehicle_config=dict(
            show_navi_mark=False, show_dest_mark=False, show_lidar=False,
            show_lane_line_detector=False, show_side_detector=False,
        ),
        vehicle_config=dict(
            show_navi_mark=False, show_line_to_navi_mark=False, show_lidar=False,
            spawn_velocity=[TARGET_SPEED_KMH / 3.6, 0.0],
            spawn_velocity_car_frame=True,
        ),
    )
    return SimpleTrackEnv(config)


def load_reference_path():
    """加载参考路径"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, 'simple_track_reference_clean.json')
    with open(path, 'r') as f:
        data = json.load(f)
    return np.array(data['reference_path']['x']), np.array(data['reference_path']['y'])


def run_experiment(exp, ref_x, ref_y):
    """运行单个实验，返回轨迹数据"""
    env = make_env()
    env.reset(seed=0)

    # 创建 PathFollower 并覆盖参数
    follower = PathFollower(ref_x, ref_y)
    follower.steer_pid = PIDController(
        kp=exp['kp_steer'], ki=exp['ki_steer'], kd=exp['kd_steer'], integral_max=1.0
    )
    follower.look_dist_base = exp['look_dist_base']
    follower.look_dist_scale = exp['look_dist_scale']
    follower.max_steer = exp['max_steer']

    origin = np.array(env.agent.position[:2], dtype=float)
    lap_counter = LapCounter(origin)
    prev_laps = 0
    dt = 0.1

    traj_x, traj_y, lat_errors = [], [], []
    max_lat_err = 0.0
    sum_lat_err = 0.0

    for frame in range(FRAMES_PER_EXP):
        vx = float(env.agent.position[0])
        vy = float(env.agent.position[1])
        vh = float(env.agent.heading_theta)
        vs = float(env.agent.speed_km_h)

        action = follower.compute_control(vx, vy, vh, vs, dt)
        env.step(action)

        traj_x.append(vx)
        traj_y.append(vy)

        laps = lap_counter.update(env.agent.position)
        if laps > prev_laps:
            env.agent.reset_navigation()
            prev_laps = laps

        le = abs(follower.log_data[-1]['lateral_error']) if follower.log_data else 0.0
        lat_errors.append(le)
        max_lat_err = max(max_lat_err, le)
        sum_lat_err += le

    env.close()

    return {
        'traj_x': traj_x,
        'traj_y': traj_y,
        'lat_errors': lat_errors,
        'laps': prev_laps,
        'max_lat_err': max_lat_err,
        'avg_lat_err': sum_lat_err / max(len(traj_x), 1),
    }


def main():
    experiments = load_params()
    ref_x, ref_y = load_reference_path()

    print(f"共 {len(experiments)} 个实验，每个 {FRAMES_PER_EXP} 帧")
    print("=" * 60)

    results = []
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {exp['id']}: {exp['description']}")
        print(f"  kp={exp['kp_steer']}, ki={exp['ki_steer']}, kd={exp['kd_steer']}, "
              f"look={exp['look_dist_base']}, scale={exp['look_dist_scale']}, "
              f"max_steer={exp['max_steer']}")

        result = run_experiment(exp, ref_x, ref_y)

        # 保存轨迹
        traj_file = os.path.join(TRAJ_DIR, f"{exp['id']}.npz")
        np.savez(traj_file,
                 traj_x=np.array(result['traj_x']),
                 traj_y=np.array(result['traj_y']),
                 lat_errors=np.array(result['lat_errors']),
                 ref_x=ref_x, ref_y=ref_y)

        results.append({'experiment_id': exp['id'], 'description': exp['description'],
                        'kp_steer': exp['kp_steer'], 'ki_steer': exp['ki_steer'],
                        'kd_steer': exp['kd_steer'], 'look_dist_base': exp['look_dist_base'],
                        'look_dist_scale': exp['look_dist_scale'], 'max_steer': exp['max_steer'],
                        'laps': result['laps'],
                        'max_lat_err': result['max_lat_err'],
                        'avg_lat_err': result['avg_lat_err']})
        print(f"  ✓ 完成: {result['laps']} 圈, "
              f"max_err={result['max_lat_err']:.2f}m, "
              f"avg_err={result['avg_lat_err']:.2f}m")
        print(f"  ✓ 轨迹已保存: {traj_file}")

    # 保存汇总结果
    summary_file = os.path.join(SCRIPT_DIR, 'results_summary.csv')
    with open(summary_file, 'w', newline='') as f:
        fieldnames = ['experiment_id', 'description', 'kp_steer', 'ki_steer', 'kd_steer',
                      'look_dist_base', 'look_dist_scale', 'max_steer',
                      'laps', 'max_lat_err', 'avg_lat_err']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\n✓ 汇总结果已保存: {summary_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
