# waypoint_pipeline

专家在 SimpleTrack 上跑 5 圈（40/48/56/64/72 km/h），10 Hz 录遥测、2 Hz 录图像，
生成「未来 3 s 共 6 点」的 ego 系 waypoints 标签；训练「单帧图像 + 车速 → waypoints」模型，
再交给几何控制器在仿真里闭环跑圈。

两个 2 Hz 不是一回事：图像 2 Hz 是输入采样率（0.5 s 一张），waypoints 2 Hz 是标签分辨率
（0.5 s 一个点）。遥测/定位始终 10 Hz。

当前数据冻结为 **baseline-v2**（7,274 图），v1（5 Hz）在 `output/archive/` 只读归档。
接手数据生产与训练，直接看 [HANDOFF.md](HANDOFF.md)；实现细节看 [方案.md](方案.md)。

---

## 1. 功能与代码

| 功能 | 文件 | 实现 |
|---|---|---|
| 多圈变速采集 | `collect_trajectory.py` | `--laps 5 --speeds 40,48,56,64,72`，每圈换目标速度，10 Hz 全量遥测，碰撞即停 |
| 图像 2 Hz 采集 | `collect_trajectory.py` + `traj_utils.capture_rgb` | 每 5 个仿真步（0.5 s）离屏抓一帧 640×360 JPEG q90，写入 `image_path` |
| 6 点时间 waypoints | `make_waypoints.py` | 在 (t,x,y) 上按 t+0.5…t+3.0 线性插值，转 ego 系；圈尾多录 6 s，仅最后 30 帧直线延拓 |
| 按圈划分 | `make_waypoints.py` / `training/common.py` | `train_mask/val_mask`，`Frames.lap_split()` 只返回带图帧，默认留最后一圈验证 |
| 几何控制器 | `controllers.py` | Pure Pursuit 取前瞻点反解前轮角 + 速度 PID + 曲率前瞻限速，10 Hz，输出 `[steering, throttle_brake]` |
| 控制器自证 | `validate_controller.py` | 喂 GT waypoints 整圈跑，检查采样相位恒 0.5 s；70 km/h 平均横向 0.02 m、0 碰撞 |
| 基线冻结校验 | `baseline_check.py` | 93 项不变量 + 4 项 sha256；`--baseline v1 / v2`、`--write-manifest`、`--allow-drift` |
| 训练栈 | `training/` | 提 DINOv3 特征 → 训 MLP head → 离线预测 → 闭环；TinyCNN 供无 GPU 冒烟 |
| 离线诊断图 | `training/report.py` | 一张 `reports/vN/summary.png`：曲线 + 分点 RMSE + 样本 + 闭环 + 自动诊断 |
| 文档体系 | `BASELINE.md` / `TRAINING_PLAN.md` / `PROJECT_LOCATION.md` / `ITERATION_LOG.md` | 基线、实验、定位、迭代日志 |

---

## 2. 用法

全部命令在 `waypoint_pipeline/` 下执行（第 0 步除外）。

### 0. 环境

```bash
cd auto-drive-1
./install.sh                                   # conda 环境 autodrive + MetaDrive + 资源
conda activate autodrive
pip install -r waypoint_pipeline/training/requirements.txt
bash components/dinov3/download_weights.sh     # 约 82 MB
```

本机 macOS 无 GPU 时：仿真用 `/usr/bin/python3`，训练用 `../.venv-train/bin/python`。
**采集不能用 `nohup`**，否则拿不到 OpenGL 上下文。

### 1. 采数据（约 4 分钟）

```bash
python collect_trajectory.py --laps 5 --speeds 40,48,56,64,72 \
    --images --out output/trajectory_5laps.json
```

得到 `output/trajectory_5laps.json`（18.4 MB）与 `output/images_trajectory_5laps/`（7,274 张，268 MB）。
`--image-interval 0.2` 可回 5 Hz，`--image-format png` 可无损。

### 2. 建集 + 冻结校验

```bash
python make_waypoints.py --traj output/trajectory_5laps.json \
    --out output/dataset_waypoints_5laps
python baseline_check.py --baseline v2 --write-manifest
```

产物 `dataset_waypoints_5laps.npz`（3.2 MB）+ `_meta.json` + `baseline_manifest.json`。
先看控制台逐圈 `|wp0|` 中位是否 ≈ `0.5 s × 均速`，再看校验是否 `97/97 项通过`。

### 3. 提特征 + 训练（GPU）

```bash
python training/extract_features.py --npz output/dataset_waypoints_5laps.npz --device cuda
python training/train.py --backbone dinov3 --device cuda --val-laps 5
```

产物：`*_features_dinov3.npz`（7,274×384）、`output/training/dinov3_best.pt`、
`dinov3_history.json`、`dinov3_split.json`。模型按验证圈 ADE 选最优。

变体：五折 `--val-laps k`、车速消融 `--zero-prop`、数据密度 `--train-stride 2`。
无 GPU 冒烟：`python training/train.py --backbone tinycnn --epochs 5 --limit-train 160 --device cpu`。

### 4. 预测 / 闭环 / 报告

```bash
python training/predict.py --ckpt output/training/dinov3_best.pt --device cuda
python training/closed_loop.py --ckpt output/training/dinov3_best.pt --speed 56 --laps 1 --device cuda
python training/report.py --version auto --title "E1 DINOv3" --ckpt output/training/dinov3_best.pt
```

判据：离线 ADE < 3 m、FDE < 5 m；闭环完圈、0 碰撞、平均横向 < 1.5 m。
闭环需要 MetaDrive；训练机没装仿真就回数据机跑。

---

## 3. 数据与接口

| 项 | 值 |
|---|---|
| 规模 | 36,371 帧遥测（3,637.1 s）/ 7,274 张图 / 0 碰撞 / 447 出界帧 |
| 划分 | 训练 1–4 圈（带图 6,178）/ 验证第 5 圈（带图 1,096） |
| 标签 | `wp_ego [N,6,2]`，t+0.5…t+3.0，ego 系（x 前，y 左，m） |
| 归一化 | `prop = [speed/72, steering/0.6]`，固定常数 |

```python
import sys; sys.path.insert(0, "waypoint_pipeline/training")
from common import Frames
f = Frames("waypoint_pipeline/output/dataset_waypoints_5laps.npz")
tr, va, laps = f.lap_split([5])   # 只返回带图帧
```

npz 18 个字段、`Frames`/`ImageDataset`/`FeatureDataset` 接口、全部配置项见
[HANDOFF.md](HANDOFF.md) §3 与 §6。

---

## 4. 配置在哪

- **数据规格**（时域 3 s、标签 2 Hz、图像 0.5 s、圈速、相机）：`config.py`，已冻结；
- **训练超参**（lr 3e-4、300 epochs、batch 64、损失权重 `3,2.5,2,1.5,1,0.8`）：`training/train.py` CLI；
- **控制器参数**（前瞻 6.0+0.35v、限幅 0.6、PID 0.15/0.02/0.01）：`config.GEO_PARAMS`。

改任一冻结项 = 新立 baseline，见 [BASELINE.md](BASELINE.md) §8。

---

## 5. 文档索引

| 文档 | 内容 |
|---|---|
| **README.md**（本文件） | 功能、用法、接口速查 |
| [HANDOFF.md](HANDOFF.md) | **交接文档**：自己生产数据 + 训练 + 全部配置 + 验收 |
| [方案.md](方案.md) | 实现方式：每个选择、关键数值、怎么配 |
| [BASELINE.md](BASELINE.md) | baseline-v1/v2 定义、sha256、不变量、变更政策 |
| [TRAINING_PLAN.md](TRAINING_PLAN.md) | 实验矩阵 E0–E7、判据、预算、风险 |
| [PROJECT_LOCATION.md](PROJECT_LOCATION.md) | 与论文/开源的阶段定位（当前 L1.5） |
| [ITERATION_LOG.md](ITERATION_LOG.md) | 每版一张诊断图 + 点评（内部） |
| [Controller验证方案.md](Controller验证方案.md) | 控制器时序验证要求与结论 |

---

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| `No module named metadrive` | `conda activate autodrive` |
| `Could not open window` | 不要用 `nohup`；用正常终端 |
| `未找到 DINOv3 权重` | `bash components/dinov3/download_weights.sh` |
| `requires torchvision` | `pip install torchvision` |
| `缺少 DINOv3 特征缓存` | 先跑 `extract_features.py` |
| `特征缓存按圈划分为空` | `--val-laps` 圈号不存在（只有 1..5） |
| `baseline_check.py` FAIL | config 被改或文件被替换；对照 BASELINE.md §3/§4 |
| 图片全读不到 | npz 与图片目录要放在一起（`image_path` 是相对路径） |
| `report.py` 中文变方块 | 装中文字体（`fonts-noto-cjk`），不影响指标 |
| 闭环跑不动 | 先用 `validate_controller.py` 排除控制器问题 |

---

## 7. 状态与限制

已完成：数据录制/建集/冻结校验、控制器整圈自证、训练栈全链路（CPU 冒烟 + DINOv3 前向）。
待做：GPU 上跑正式精度实验（E1–E7），见 TRAINING_PLAN。

限制：单帧无历史；无交通/动态障碍；72 km/h 是未见过速度；闭环图与训练图同管线但未做域随机化。