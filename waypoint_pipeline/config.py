"""waypoint_pipeline 全局配置：时间 waypoints 数据制作 + waypoints 控制器。

约定:
- 仿真控制频率 10 Hz (decision_repeat=5 x physics 0.02s = 0.1s/step)
- waypoints: 未来 5 秒, 2 Hz, 共 10 个点, ego 坐标系 (x 前, y 左), 单位米
- 所有运行均为无头模式 (use_render=False), 不弹出任何窗口
"""
from __future__ import annotations

from pathlib import Path

# ---- 路径 ----
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
REF_PATH_FILE = PROJECT_ROOT / "simple_track_reference_clean.json"
OUT_DIR = HERE / "output"

# ---- 时间 waypoints 规格 ----
WP_HORIZON_S = 5.0        # 未来时域
WP_FREQ_HZ = 2.0          # 采样频率
WP_COUNT = int(WP_HORIZON_S * WP_FREQ_HZ)   # 10 个点
WP_DT = 1.0 / WP_FREQ_HZ                    # 0.5 s

# ---- 仿真 / 采集 ----
SIM_DT = 0.1              # 每个 env.step 的仿真时长 (10 Hz)
RECORD_HZ = 10            # 轨迹记录频率, 与 SIM_DT 一致
TARGET_SPEED_KMH = 70.0   # 与 simple_track.TARGET_SPEED_KMH 一致
TAIL_EXTRA_S = 6.0        # 过终点后多录几秒, 保证末尾帧也有完整未来 5s
QUICK_SECONDS = 90.0      # quick 模式只采前 90 秒 (直道+第一个大弯)

# ---- 图片采集 (run_pipeline --images, 离屏渲染不弹窗) ----
IMAGE_INTERVAL_S = 2.0                    # 每 2s 采一帧 (0.5 Hz)
IMAGE_EVERY_STEPS = int(IMAGE_INTERVAL_S / SIM_DT)  # = 20 步
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 360
IMAGE_FOV_DEG = 60.0                      # 水平 FOV (与 viz_waypoints 一致)
IMAGE_DIR_NAME = "images"                 # output/images/

# ---- 几何控制器 (Pure Pursuit + PID) 默认参数 ----
# 注意: MetaDrive 动作 steering ∈ [-1,1] 归一化, 实际前轮角 = 动作 × max_steering(40°)
# 实测标定: 小转向(≤0.1)自行车模型准确; 大转向轮胎饱和, 横摆率封顶 ~0.5rad/s,
# 70km/h 下最小转弯半径 ~39m。急弯需要大转向指令, 专家幅值可达 ±0.6。
GEO_PARAMS = dict(
    look_base=6.0,        # 基础前瞻距离 m
    look_scale=0.35,      # 前瞻随速系数 s (L = base + scale * v)
    look_min=4.0,
    look_max=28.0,
    kp_steer=0.0,         # 横向修正增益: ego系waypoints下默认关闭(避免重复计数)
    max_steer=0.6,        # 归一化动作限幅 (与专家 PathFollower 一致, 急弯需要)
    max_steering_deg=40.0,  # MetaDrive 车辆 max_steering: 动作×40°=前轮角
    wheelbase=2.5,        # m
    kp_speed=0.15,
    ki_speed=0.02,
    kd_speed=0.01,
    integral_max=3.0,
    lat_acc_max=4.5,      # 弯道限速: v <= sqrt(a_lat/|kappa|); 专家 R100 弯不减速(≈3.8m/s²)
    curve_preview_s=3.0,  # 用未来 3s 内的最大曲率限速
)

# ---- 评测判据 ----
EVAL_LATERAL_WARN_M = 1.5     # 平均横向误差警戒值
EVAL_OFFROAD_MAX_RATIO = 0.01  # 出界帧占比上限