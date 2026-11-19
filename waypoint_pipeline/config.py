"""waypoint_pipeline 全局配置：时间 waypoints 数据制作 + waypoints 控制器。

约定:
- 仿真控制频率 10 Hz (decision_repeat=5 x physics 0.02s = 0.1s/step)
- waypoints(标签): 未来 3 秒, 2 Hz, 共 6 个点, ego 坐标系 (x 前, y 左), 单位米
  时域取 3 s 而不是 5 s: 更远的点对控制没有作用, 且弯道上转角更大,
  拟合 loss 更高反而误导模型; 3 s @ 70 km/h 已覆盖 ~58 m 前瞻。
- 图像(输入): 采集频率 2 Hz (0.5 s 一张)。
  【不要和标签的 2 Hz 混为一谈】两者数值相同但作用在不同端:
    * WP_FREQ_HZ 定义“未来预测多密”——t+0.5 … t+3.0 共 6 个点;
    * IMAGE_INTERVAL_S 定义“多久看一次”——决定训练样本的时间密度。
  遥测/定位始终 10 Hz 全量, 不受图像频率影响。
- 所有运行均为无头模式 (use_render=False), 不弹出任何窗口
"""
from __future__ import annotations

from pathlib import Path

# ---- 路径 ----
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
REF_PATH_FILE = PROJECT_ROOT / "simple_track_reference_clean.json"
OUT_DIR = HERE / "output"

# ---- 基线标识 (冻结) ----
# Baseline v1 = 专家 PathFollower 沿参考线正常跑 5 圈 (40/48/56/64/72 km/h),
# 无扰动/无障碍/无交通/无噪声, 固定 seed=0; 图像 5 Hz。已归档, 只读。
# Baseline v2 = 与 v1 同专家/同圈速/同场景, 唯一差异是图像采集 5 Hz → 2 Hz。
# 好坏如实保留, 不做任何人为调整。
# 任何数据规格、采集方式、圈速的改动都必须新建 baseline id (v3...), 不得原地修改已有版本。
# 校验: python3 baseline_check.py (不变量 + sha256), 说明见 BASELINE.md
BASELINE_ID = "baseline-v2"

# ---- 时间 waypoints 规格 ----
WP_HORIZON_S = 3.0        # 未来时域 (3 s, 控制够用; 更远只会加大 loss)
WP_FREQ_HZ = 2.0          # 采样频率
WP_COUNT = int(WP_HORIZON_S * WP_FREQ_HZ)   # 6 个点
WP_DT = 1.0 / WP_FREQ_HZ                    # 0.5 s

# ---- 仿真 / 采集 ----
SIM_DT = 0.1              # 每个 env.step 的仿真时长 (10 Hz)
RECORD_HZ = 10            # 轨迹记录频率, 与 SIM_DT 一致
TARGET_SPEED_KMH = 70.0   # 单圈采集默认速度; 与 simple_track.TARGET_SPEED_KMH 一致
TAIL_EXTRA_S = 6.0        # 过终点后多录几秒, 保证末尾帧也有完整未来 3s
QUICK_SECONDS = 90.0      # quick 模式只采前 90 秒 (直道+第一个大弯)

# ---- 多圈变速采集 (训练集) ----
# 每圈一个巡航速度, 速度跨度 8 km/h, 用于检验模型对不同车速的泛化。
# 上限 72 km/h: MetaDrive 默认 max_speed_km_h=80, 留出 PID 调节余量。
N_LAPS = 5
LAP_SPEEDS_KMH = (40.0, 48.0, 56.0, 64.0, 72.0)
MULTI_LAP_TRAJ = OUT_DIR / "trajectory_5laps.json"
MULTI_LAP_DATASET = OUT_DIR / "dataset_waypoints_5laps"   # 输出前缀 (不带扩展名)

# ---- 图片采集 (run_pipeline --images, 离屏渲染不弹窗) ----
# 遥测/定位始终按 10 Hz 全量记录; 图片 2 Hz (0.5 s 一帧)。
# 需要更密可传 --image-interval 0.2, 更稀可传 1.0。
# JPEG 90 质量约 50-80 KB/帧, 5 圈 2 Hz 约 0.3 GB。
# 注意: 这是【输入】采样率, 与 WP_FREQ_HZ(【标签】采样率) 是两件事。
IMAGE_INTERVAL_S = 0.5                    # 每 0.5s 采一帧 (2 Hz)
IMAGE_EVERY_STEPS = int(IMAGE_INTERVAL_S / SIM_DT)  # = 5 步
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 360
IMAGE_FOV_DEG = 60.0                      # 水平 FOV (与 viz_waypoints 一致)
IMAGE_DIR_NAME = "images"                 # output/images/
IMAGE_FORMAT = "jpg"                      # jpg (推荐, 体积小) / png
IMAGE_QUALITY = 90                        # JPEG 质量

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

# ---- 训练 / 推理 (training/) ----
TRAIN_DIR = HERE / "training"
TRAIN_OUT_DIR = OUT_DIR / "training"      # checkpoint / history / 预测图

# ---- 评测判据 ----
EVAL_LATERAL_WARN_M = 1.5     # 平均横向误差警戒值
EVAL_OFFROAD_MAX_RATIO = 0.01  # 出界帧占比上限