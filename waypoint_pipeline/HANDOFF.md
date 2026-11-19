# 交接文档 — 数据生产 + 模型训练

> 交接对象：接手本项目的工程师 / Agent。
> 交接物：**代码 + 文档**。数据**不随代码交付**（`output/` 在 `.gitignore` 里），
> 训练数据必须由接收方在本地用 MetaDrive **现场生产**（见 §2）。
>
> 你负责四件事：生产数据 → 提特征 → 训练 → 评估（闭环可选）。
> 交付边界：数据规格已冻结为 `baseline-v2`，改规格 = 新立基线，见 §9。

---

## 0. 一眼看清要做什么

| 步骤 | 命令 | 产物 | 耗时（参考） |
|---|---|---|---|
| 0 环境 | `./install.sh` + `pip install -r training/requirements.txt` | conda 环境 | 10–20 分钟 |
| 1 采数据 | `collect_trajectory.py --laps 5 ... --images` | 轨迹 JSON + 7,274 张图 | 约 4–6 分钟（wall） |
| 2 建集 | `make_waypoints.py` | `.npz` 3.2 MB + meta | < 10 秒 |
| 3 冻结校验 | `baseline_check.py --baseline v2 --write-manifest` | 97 项 PASS | 约 20 秒 |
| 4 提特征 | `extract_features.py --device cuda` | 7,274 × 384 特征缓存 | GPU 约 1 分钟 |
| 5 训练 | `train.py --backbone dinov3` | `dinov3_best.pt` | GPU 数分钟 |
| 6 评估 | `predict.py` / `closed_loop.py` | 指标 JSON + 图 | 1 分钟 / 每圈数分钟 |
| 7 报告 | `report.py` | `reports/vN/summary.png` | < 1 分钟 |

所有命令都在 `waypoint_pipeline/` 下执行（除第 0 步）。

---

## 1. 环境准备

### 1.1 两套 Python 环境

| 用途 | 依赖 | 说明 |
|---|---|---|
| 生产数据 / 闭环 | `metadrive-main[gym]` + `panda3d` + numpy + Pillow | 需要 OpenGL 离屏渲染；**不能用 `nohup`**，否则拿不到 GL 上下文 |
| 训练 / 离线评估 | torch ≥ 2.2、torchvision ≥ 0.17、transformers ≥ 4.56、Pillow、numpy、matplotlib | 不需要 MetaDrive（`closed_loop.py` 除外） |

```bash
cd auto-drive-1
./install.sh                                    # 建 conda 环境 autodrive，装 MetaDrive + 资源
conda activate autodrive
pip install -r waypoint_pipeline/training/requirements.txt
bash components/dinov3/download_weights.sh      # ViT-S/16 权重，约 82 MB（ModelScope）
```

国内网络可加 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`。
没有 `conda` 时：`pip install -e metadrive-main[gym]` + `python -m metadrive.pull_asset`（下载 3D 资源）。

### 1.2 目录要求（相对位置不能变）

```
auto-drive-1/
├── simple_track.py / runtime_config.py / pid_demo_autonomous.py  # 专家 + 地图
├── simple_track_reference_clean.json                             # 整圈参考线
├── metadrive-main/                                               # 仿真器（含 assets）
├── components/dinov3/                                            # 冻结编码器 + 权重
└── waypoint_pipeline/
    ├── config.py            # 唯一事实源（规格、圈速、图像、控制器参数）
    ├── controllers.py       # 下游控制器（闭环要用）
    ├── traj_utils.py        # 无头环境 / 抓帧 / 轨迹读写
    ├── training/            # 训练栈（可单独拷到训练机）
    └── output/              # 你生产的数据与 checkpoint 落这里
```

`training/common.py` 会自动向上两级找项目根并注入 `sys.path`，所以
**训练机最小集** = `waypoint_pipeline/{config.py, controllers.py, traj_utils.py, training/}` + `components/dinov3/`；
**生产数据必须是完整仓库**（要用专家、地图、参考线与资源）。

---

## 2. 生产数据（不能拷贝，必须自己跑）

### 2.1 为什么必须自己生产

- 数据是仿真产物，依赖 MetaDrive 版本、assets 与渲染后端；
- 本机（macOS）与训练机（Linux）渲染出来的图像不保证逐字节相同；
- 只有现场生成，`baseline_check.py` 的不变量和 sha256 才是对本机数据的有效冻结。
  仓库里的 `output/archive/baseline-v1-hz5/` 与 baseline-v2 的校验值只作**规格对照**，
  不能当作训练输入（你也拿不到）。

### 2.2 三条命令

```bash
cd waypoint_pipeline

# 1) 5 圈变速录制：遥测/定位 10 Hz 全量；图像 2 Hz（0.5 s 一张）
python collect_trajectory.py --laps 5 --speeds 40,48,56,64,72 \
    --images --out output/trajectory_5laps.json

# 2) 轨迹 → 未来 3 s @ 2 Hz 的 6 点 ego waypoints 数据集
python make_waypoints.py --traj output/trajectory_5laps.json \
    --out output/dataset_waypoints_5laps

# 3) 冻结校验 + 写本机清单（首次生产时执行）
python baseline_check.py --baseline v2 --write-manifest
```

关键参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--laps` | 1 | 生产训练集用 **5** |
| `--speeds` | 单值 70 | 每圈一个巡航速度，逗号分隔；本方案 `40,48,56,64,72` |
| `--images` | 关 | 打开相机采集（不开就没有训练输入） |
| `--image-interval` | 0.5 | 来自 `config.IMAGE_INTERVAL_S`；0.2 即 5 Hz（体积 ×2.5） |
| `--image-format` | jpg | `jpg`（q90，约 40 KB/张）或 `png` |
| `--image-dir` | `output/images_<out_stem>/` | 图片目录 |
| `--out` | `output/trajectory_expert.json` | 轨迹 JSON 路径 |
| `--quick` | 关 | 只采前 90 s，仅调试用 |

### 2.3 验收标准（对照 baseline-v2）

控制台 + `dataset_waypoints_5laps_meta.json` 逐项核对：

| 项 | 期望值 | 容差 |
|---|---|---|
| 遥测帧数 | 36,371（10 Hz × 3,637.1 s） | ±2% |
| 图像张数 | 7,274（2 Hz） | ±2% |
| 碰撞帧 | 0 | 必须 |
| 出界帧 | 447（1.23%，专家自身轻微出界） | ≤2% |
| 仿真时长 | 3,637.1 s | ±2% |
| 建集间距相对误差 | 0.0001 | < 0.10 |
| 逐圈 `\|wp0\|` 中位 | 5.556 / 6.667 / 7.778 / 8.889 / 10.0 m | ±0.15 m |

逐圈参考（5 圈、每圈一个速度）：

| 圈 | 速度 | 圈时 | 帧数 | 图片 |
|---|---|---|---|---|
| 1 | 40 km/h | 971.3 s | 9,713 | 1,942 |
| 2 | 48 km/h | 812.4 s | 8,124 | 1,625 |
| 3 | 56 km/h | 696.3 s | 6,963 | 1,393 |
| 4 | 64 km/h | 609.4 s | 6,094 | 1,218 |
| 5 | 72 km/h | 541.6 s | 5,477 | 1,096 |

**判定**：`baseline_check.py` 退出码 0，末尾打印 `[baseline] 97/97 项通过` 且无失败项。
若 MetaDrive 版本与生成 baseline-v2 时不同，可加 `--allow-drift`（计数 ±2%、`|wp0|` ±0.15 m），
它只证明"方案一致"，sha256 以你本机新写的 `baseline_manifest.json` 为准。

### 2.4 产物目录

```
waypoint_pipeline/output/
├── trajectory_5laps.json              # 18.4 MB，每帧全遥测 + 图片相对路径
├── images_trajectory_5laps/           # 7,274 张 JPEG，640×360，约 268 MB
├── dataset_waypoints_5laps.npz        # 3.2 MB，训练集（18 个字段，见 §3.1）
├── dataset_waypoints_5laps_meta.json  # 数据体检报告
└── baseline_manifest.json             # 冻结清单（不变量 + sha256）
```

`image_path` 存的是**相对 npz 所在目录**的路径：**npz 与图片目录必须一起搬**，
单独拷 npz 会导致所有图片读不到。

---

## 3. 数据接口（训练代码读什么）

### 3.1 `dataset_waypoints_5laps.npz` 字段（18 个）

| 字段 | 形状 | 说明 |
|---|---|---|
| `wp_ego` | [N,6,2] f32 | **标签**：t+0.5 … t+3.0，ego 系（x 前，y 左，m） |
| `wp_world` | [N,6,2] f32 | 世界系（可视化/调试） |
| `t,x,y,heading` | [N] | 时间与位姿 |
| `speed_kmh` | [N] f32 | 当前车速 |
| `steering,throttle,brake` | [N] | 专家控制量 |
| `lap,lap_t` | [N] int32/f32 | 圈号 1..5、圈内时间 |
| `target_speed_kmh` | [N] f32 | 该圈目标速度 |
| `yaw_rate,accel_ms2` | [N] f32 | 横摆角速度、纵向加速度 |
| `image_path` | [N] `<U128` | 相对 npz 目录；空串 = 该帧无图 |
| `train_mask,val_mask` | [N] bool | 参考划分（多圈=留最后一圈验证） |

### 3.2 Python 接口

```python
import sys
sys.path.insert(0, "waypoint_pipeline/training")

from common import Frames, make_prop
from dataset import ImageDataset, FeatureDataset, load_feature_cache, split_feature_cache

f = Frames("waypoint_pipeline/output/dataset_waypoints_5laps.npz")
f.wp.shape            # (36371, 6, 2)
f.image_mask.sum()    # 7274，有图的帧
tr, va, laps = f.lap_split([5])      # 只返回带图帧；默认 val = 最后一圈
img_path = f.image_abs(int(tr[0]))   # 绝对路径 = npz 目录 / image_path
prop = make_prop([f.speed[tr[0]]], [f.steering[tr[0]]])   # [[speed/72, steer/0.6]]
```

| 对象 | 用途 |
|---|---|
| `Frames` | npz 轻量视图：`wp/speed/steering/t/lap/lap_t/yaw_rate/image`、`image_abs()`、`lap_split()`、`summarize()` |
| `ImageDataset` | 原始图像 → 张量（缩放 + 可选水平翻转），TinyCNN 用 |
| `FeatureDataset` | 预提取 384 维特征 + proprio + GT，加高斯噪声/特征 dropout |
| `load_feature_cache` / `split_feature_cache` | 读 `*_features_dinov3.npz`，按 `src_index` 映射回圈号并划分 |

### 3.3 归一化（固定常数，不得改）

```python
prop = [speed_kmh / 72.0, steering / 0.6]
```

`72` 是采集速度上限、`0.6` 是专家转向幅值上限（见 `config.GEO_PARAMS["max_steer"]`）。
用固定常数而不是训练集统计量，是为了让推理端不需要带均值方差。

---

## 4. 训练

### 4.1 自检（不需要数据 / GPU）

```bash
python training/selftest.py       # 期望最后一行 SELFTEST PASS
```

### 4.2 提取 DINOv3 冻结特征

```bash
python training/extract_features.py \
    --npz output/dataset_waypoints_5laps.npz --device cuda
```

- 只处理有图帧（7,274）；模型冻结，特征 = `(CLS + patch 均值) / 2`，384 维；
- 输出 `output/dataset_waypoints_5laps_features_dinov3.npz`：
  `features[M,384]`、`prop[M,2]`、`wp[M,6,2]`、`src_index[M]`（原 npz 帧号）、
  `speed_kmh/lap/t/image_path`、`source_npz`；
- CPU 约 60–70 帧/s（2–5 分钟），GPU 约 1 分钟；
- 参数：`--out`（默认同名 `_features_dinov3.npz`）、`--batch-size 32`、`--limit N`、`--seed 0`。

### 4.3 训练 head

```bash
# 主实验 E1：第 1–4 圈训练，第 5 圈（72 km/h，未见速度）验证
python training/train.py --backbone dinov3 --device cuda --val-laps 5

# 五折速度泛化 E2
for k in 1 2 3 4 5; do
  python training/train.py --backbone dinov3 --device cuda --val-laps $k \
      --out-dir output/training/fold$k
done

# 车速消融 E3（proprio 置 0）
python training/train.py --backbone dinov3 --device cuda --val-laps 5 \
    --zero-prop --out-dir output/training/no_prop

# 数据密度消融 E5（每 2 帧取 1）
python training/train.py --backbone dinov3 --device cuda --val-laps 5 \
    --train-stride 2 --out-dir output/training/stride2
```

训练逻辑（`train.py`）：

- 划分：`lap_split` 按圈 hold-out，默认验证圈 = 最后一圈；相邻帧不跨集；
- 损失：加权 `SmoothL1(beta=1)`，点权重 `[3.0,2.5,2.0,1.5,1.0,0.8]`（近处权重大）
  \+ `0.02 ×` 二阶差分平方（waypoint 序列平滑）；
- 优化：AdamW，lr `3e-4`，weight decay `1e-4`，Cosine 退火到 `lr×0.02`，
  batch 64，梯度裁剪 5.0，最多 300 epochs（`--patience` 可早停）；
- 模型选择：按**验证圈 ADE** 保存最优，不是按 loss。

### 4.4 产物

| 文件 | 内容 |
|---|---|
| `output/training/dinov3_best.pt` | 最优权重 + `config`（backbone/wp_count/feat_dim/hidden_dim/image_size/prop_dim）+ `val` + `val_laps` + `loss_weights` + `smooth_weight` + `zero_prop` + `feature_cache` |
| `output/training/dinov3_history.json` | 每 epoch 的 train_loss / val loss / ADE / FDE / 分点 RMSE / lr |
| `output/training/dinov3_split.json` | 数据路径、训练/验证样本数、最优 epoch 与 ADE |

预算：head 可训练参数约 0.17 M；6,178 训练样本、384 维特征，GPU 上 300 epochs 数分钟。
无 GPU 时用 TinyCNN 冒烟：`--backbone tinycnn --epochs 5 --limit-train 160 --device cpu`
（`--npz output/dataset_waypoints.npz` 旧单圈集即可，精度无意义）。

---

## 5. 评估

### 5.1 离线预测（Prediction）

```bash
python training/predict.py --ckpt output/training/dinov3_best.pt --device cuda
```

产物：`pred_metrics.json`（ADE/FDE/分点 RMSE/按圈与按速度分桶）、`pred_arrays.npz`、
`pred_per_frame.csv`（逐帧误差，用来找最差帧）、`pred_vs_gt.png`、`per_point_rmse.png`。

判据：**ADE < 3 m、FDE < 5 m**；若 FDE 明显大于 2×ADE，先查时域/数据而非换模型。

### 5.2 闭环测试（Test，可选，需要 MetaDrive）

```bash
python training/closed_loop.py --ckpt output/training/dinov3_best.pt \
    --speed 56 --laps 1 --device cuda
```

- 链路：实时渲染图 → 模型 → 6 点 waypoints → `GeometricController` → 动作，10 Hz；
- `--infer-every N`：每 N 个 tick 推理一次，中间用位姿变换复用 waypoints（CPU 可设 2–3）；
- 产物：`closed_loop_metrics.json`、`closed_loop_log.csv`、`closed_loop.png`；
- 判据：完圈、0 碰撞、`lat_abs_mean_m < 1.5`、`offroad_ratio < 1%`。

### 5.3 控制器自证（与模型无关）

```bash
python validate_controller.py             # GT waypoints 整圈
python validate_controller.py --quick     # 90 s 片段，先看时序断言
```

用于区分"模型问题"和"控制器问题"。已通过：70 km/h 整圈、平均横向 0.02 m、0 碰撞。

### 5.4 离线单图诊断（每个版本必做）

```bash
python training/report.py --version auto --title "E1 DINOv3 1-4圈" \
    --ckpt output/training/dinov3_best.pt
```

产物 `reports/<vN>/summary.png` + `summary_metrics.json`，并同步 `reports/latest/`。
一张图含训练曲线、分点 RMSE、样本（最好/中位/最差）、ADE×车速、ADE×弯道、闭环轨迹与时序、自动诊断。
版本规则：任何代码/超参/评测改动都要新开 `vN+1`，图与点评追加进 `ITERATION_LOG.md`；对外只交付 latest。

---

## 6. 全部配置

### 6.1 `config.py` 冻结项（改了就不是 baseline-v2）

| 项 | 值 | 含义 |
|---|---|---|
| `BASELINE_ID` | `baseline-v2` | 基线标识 |
| `WP_HORIZON_S` | 3.0 | 标签时域 3 s |
| `WP_FREQ_HZ` | 2.0 | 标签采样 2 Hz（0.5 s 间隔） |
| `WP_COUNT` / `WP_DT` | 6 / 0.5 | 点数 / 点间隔 |
| `SIM_DT` / `RECORD_HZ` | 0.1 / 10 | 控制与遥测 10 Hz |
| `TAIL_EXTRA_S` | 6.0 | 过终点多录 6 s（> 3 s 时域） |
| `N_LAPS` | 5 | 采集圈数 |
| `LAP_SPEEDS_KMH` | (40,48,56,64,72) | 每圈巡航速度，步长 8 |
| `TARGET_SPEED_KMH` | 70.0 | 单圈默认速度（多圈由 `--speeds` 覆盖） |
| `IMAGE_INTERVAL_S` | 0.5 | **输入**采样 2 Hz（与标签 2 Hz 不是一回事） |
| `IMAGE_EVERY_STEPS` | 5 | = 0.5 / 0.1 |
| `IMAGE_WIDTH×HEIGHT` | 640×360 | 相机分辨率 |
| `IMAGE_FOV_DEG` | 60.0 | 水平视场角 |
| `IMAGE_FORMAT/QUALITY` | jpg / 90 | 压缩格式与质量 |

### 6.2 `config.py` 训练 / 评测项

| 项 | 值 | 说明 |
|---|---|---|
| `TRAIN_DIR` / `TRAIN_OUT_DIR` | `training/` / `output/training` | 代码与产物目录 |
| `MULTI_LAP_TRAJ` / `MULTI_LAP_DATASET` | `output/trajectory_5laps.json` / `output/dataset_waypoints_5laps` | 默认数据路径 |
| `EVAL_LATERAL_WARN_M` | 1.5 | 闭环平均横向误差警戒 |
| `EVAL_OFFROAD_MAX_RATIO` | 0.01 | 出界帧占比上限 |

### 6.3 `train.py` 参数（默认值即正式配方）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--npz` | 自动（优先 5 圈集） | 数据集 |
| `--backbone` | `dinov3` | `dinov3` / `tinycnn` |
| `--features` | 自动同名缓存 | DINOv3 特征缓存 |
| `--val-laps` | 最后一圈 | 验证圈号，如 `5` 或 `2,4` |
| `--out-dir` | `output/training` | checkpoint / history |
| `--epochs` | 300 | 上限 |
| `--batch-size` | 64 | |
| `--lr` / `--weight-decay` | 3e-4 / 1e-4 | AdamW |
| `--hidden-dim` | 256 | MLP 宽度 |
| `--image-size` | `128x72` | 仅 TinyCNN |
| `--loss-weights` | `3,2.5,2,1.5,1,0.8` | 6 点权重（按 wp_count 截取） |
| `--smooth-weight` | 0.02 | 二阶平滑项 |
| `--limit-train` | 0 | 只用前 N 个样本（冒烟） |
| `--train-stride` | 1 | 训练帧抽稀（消融） |
| `--zero-prop` | 关 | proprio 置 0（消融） |
| `--num-workers` | 0 | |
| `--device` | auto | `auto/cpu/cuda/mps` |
| `--seed` | 0 | |
| `--patience` | 0 | 早停轮数，0 = 关 |

### 6.4 控制器参数（`config.GEO_PARAMS`）

| 参数 | 值 | 含义 |
|---|---|---|
| `look_base` / `look_scale` | 6.0 / 0.35 | 前瞻 L = 6.0 + 0.35·v，限 [4, 28] m |
| `kp_steer` | 0.0 | 横向 P 项关闭（ego waypoints 已含偏移） |
| `max_steer` | 0.6 | 转向动作限幅（急弯需要，标定值） |
| `max_steering_deg` | 40.0 | 动作 → 前轮角 = 动作 × 40° |
| `wheelbase` | 2.5 | 轴距 m |
| `kp_speed / ki_speed / kd_speed` | 0.15 / 0.02 / 0.01 | 速度 PID |
| `integral_max` | 3.0 | 积分限幅 |
| `lat_acc_max` | 4.5 | 弯道限速 v ≤ √(a/κ) |
| `curve_preview_s` | 3.0 | 用未来 3 s 最大曲率限速 |

### 6.5 训练固定常数（写在代码里，不要改）

| 常数 | 值 | 位置 |
|---|---|---|
| `SPEED_SCALE_KMH` / `STEER_SCALE` | 72.0 / 0.6 | `training/common.py` |
| 损失点权重 | `[3.0,2.5,2.0,1.5,1.0,0.8]` | `training/train.py` |
| 平滑权重 / 梯度裁剪 | 0.02 / 5.0 | `training/train.py` |
| Dropout | 0.1 | `training/model.py` |
| 特征增强 | 噪声 σ=0.02、dropout 0.05 | `training/dataset.py` |

---

## 7. 验收清单

数据侧：

1. `baseline_check.py --baseline v2` 退出码 0（97 项 PASS）；
2. `meta.json`：`num_frames=36,371`、`num_images=7,274`、`spacing_median_rel_err<0.10`；
3. 逐圈 `|wp0|` 中位 = 0.5 s × 该圈均速；0 碰撞。

训练侧：

4. `selftest.py` PASS；
5. `extract_features.py` 输出 `[7274, 384]`；
6. `train.py` 的 split 里 `train_samples=6178`、`val_samples=1096`；
7. `predict.py` ADE < 3 m、FDE < 5 m（E1）。

评估侧：

8. 闭环每档速度完圈、0 碰撞、平均横向 < 1.5 m；
9. `reports/vN/summary.png` 已生成并写点评。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: metadrive` | 没激活 conda 环境；`conda activate autodrive` |
| `Could not open window` / GL 报错 | 用了 `nohup` 或后台无终端；改正常终端运行 |
| `未找到 DINOv3 权重` | `bash components/dinov3/download_weights.sh` |
| `DINOv3ViTImageProcessorFast requires torchvision` | `pip install torchvision` |
| `缺少 DINOv3 特征缓存` | 先跑 `extract_features.py`，或 `--features` 指定 |
| `特征缓存按圈划分为空` | `--val-laps` 传了不存在的圈号（5 圈只有 1..5） |
| `baseline_check.py` FAIL | config 被改 / 文件被替换；对照 BASELINE.md §3/§4 |
| `make_waypoints` 覆盖了单圈数据集 | 它默认输出 `dataset_waypoints`，务必传 `--out output/dataset_waypoints_5laps` |
| 训练 loss 正常但 ADE 很大 | 用了旧单圈集，或未训练完；看 `dinov3_split.json` 的样本数 |
| 闭环一开局就撞 | 模型未训练或数据链路错；先跑 `validate_controller.py` |
| CPU 闭环太慢 | `--infer-every 2` 或 3；看日志 `infer_ms` |
| `report.py` 中文变方块 | 缺中文字体，Ubuntu 装 `fonts-noto-cjk`（不影响指标） |
| 图片全部读不到 | npz 与图片目录分开放了；`image_path` 相对 npz 目录 |

---

## 9. 禁止事项（会破坏可复现性）

1. **不得改冻结规格**（时域、频率、圈速、图像规格）。要改 → 新立 `baseline-v3`、
   新目录、新校验值，旧版本只读。
2. **不得人为筛选 / 补录 / 删改数据**。好帧坏帧（出界、抖动）原样保留。
3. **不得随机划分**。必须按圈 hold-out，否则相邻帧泄漏、指标虚高。
4. **不得用旧单圈 `dataset_waypoints.npz` 做正式实验**（只用于冒烟）。
5. **不得改归一化常数**（72 / 0.6）与损失权重后还称同一个实验；属于新版本。
6. **对外只交付 `reports/latest/summary.png`**，版本图与点评留在 `ITERATION_LOG.md` 内部。

---

## 10. 文档分工

| 文档 | 看什么 |
|---|---|
| **HANDOFF.md**（本文件） | 生产数据 + 训练 + 全部配置 |
| [README.md](README.md) | 用法与命令速查 |
| [方案.md](方案.md) | 实现方式与关键数值 |
| [BASELINE.md](BASELINE.md) | 基线定义、sha256、不变量、变更政策 |
| [TRAINING_PLAN.md](TRAINING_PLAN.md) | 实验矩阵 E0–E7、判据、预算 |
| [PROJECT_LOCATION.md](PROJECT_LOCATION.md) | 与论文/开源的定位对比 |
| [ITERATION_LOG.md](ITERATION_LOG.md) | 每版诊断图 + 点评 |