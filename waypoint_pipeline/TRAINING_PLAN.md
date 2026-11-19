# Waypoint 模型：训练 / Prediction / Test 实验设计

本文件是 `waypoint_pipeline` 的训练侧实验规范。数据规格、目录与命令总览见
[README.md](README.md)。目标不是“把 loss 压低”，而是回答三个问题：

1. **单帧图像 + 车速能否预测未来 3 s 的 6 点轨迹？**（离线精度）
2. **预测的 waypoints 交给几何控制器能否安全跑完整圈？**（闭环可用性）
3. **换一个没见过的车速，性能掉多少？**（速度泛化）

---

## 1. 数据与划分

### 1.1 数据来源

本轮唯一合法数据为 **baseline-v2**（定义与校验值见 [BASELINE.md](BASELINE.md)）：
专家按原方案正常跑线，**无扰动/无障碍/无困难样本注入**，好坏如实保留；
E1–E7 全部在此数据上进行，不得为了提升指标而临时改数据或筛选帧。
（**baseline-v1** 是图像 5 Hz 的旧版，已归档只读，不可与 v2 混用。）

| 项 | 值 |
|---|---|
| 地图 | SimpleTrack 闭环（约 10.8 km） |
| 圈数 | 5 圈，逐圈目标速度 **40 / 48 / 56 / 64 / 72 km/h** |
| 遥测 | 10 Hz 全量：位姿、速度、横摆角速度、加速度、转向/油门/刹车、出界/碰撞、圈号 |
| 图像 | 640×360，FOV 60°，**2 Hz（0.5 s 一帧）**，JPEG |
| 标签 | 每帧未来 3 s @ 2 Hz 的 6 个 ego 系 waypoints（t+0.5 … t+3.0） |
| 规模 | 36,371 帧遥测（10 Hz）/ **7,274 张图（2 Hz）** / 1.01 h |

### 1.2 划分原则：按圈，不按帧随机

相邻帧（0.1 s 遥测 / 0.5 s 图像）高度相关，随机划分会把“同一段路的相邻帧”分到训练和验证，
指标虚高。默认：

- **E1 主实验**：训练第 1–4 圈（40/48/56/64），验证第 5 圈（72 km/h，未见过）。
  带图像帧数：训练 **6,178** / 验证 **1,096**。
- **E2 速度泛化**：leave-one-speed-out 五折，每折留一圈做验证，报告五折均值±标准差。
- 不推荐随机划分；如必须随机，需按时间块（例如每 30 s 取一整块）划分并在报告中注明。

### 1.3 数据质量门槛（训练前必须通过）

先跑 `python3 baseline_check.py`（冻结校验，应全部 PASS，退出码 0），再看：

- `dataset_waypoints_5laps_meta.json` 中 `spacing_median_rel_err < 0.10`；
- 每圈 `wp0_median_m ≈ 0.5 s × 实测均速`（40 km/h ≈ 5.6 m，72 km/h ≈ 10.0 m），
  `wp0_max_m` 与中位数相差 < 0.5 m；
- 碰撞帧为 0；出界率记录在案（专家自身也可能轻微出界）；
- 圈数与逐圈速度、图片数量与圈号分布一致。

---

## 2. 模型与输入

```
image 640x360 ──► DINOv3 ViT-S/16 (frozen) ──► CLS+patch mean [384] ─┐
                                                                     ├─► MLP 256 ─► 6×2 waypoints
vehicle state: speed/72, steering/0.6 ──────────────────────────────┘
```

**关键设计：必须输入车速。** waypoints 是时间采样，40 km/h 与 72 km/h 下同一点的距离
相差 1.8 倍，单帧图像无法区分；没有车速输入时，模型只能学到训练集速度均值，闭环会
系统性偏短/偏长。E3 用 `--zero-prop` 做消融验证。

**backbone 的选择**：

| backbone | 用途 | 依赖 |
|---|---|---|
| `dinov3` | 正式交付模型；冻结特征 + 轻量 head | torch + transformers + DINOv3 权重 |
| `tinycnn` | 无 GPU / 无权重时的冒烟测试，验证整条链路 | 仅 torch |

TinyCNN 的结果不作为精度结论，只证明数据、训练、预测、闭环代码可运行。

---

## 3. 训练配方（默认）

| 项 | 值 | 理由 |
|---|---|---|
| 损失 | 加权 SmoothL1 (β=1 m) + 0.02×二阶平滑 | 近点更影响控制；平滑项抑制折线抖动 |
| 点权重 | 3.0, 2.5, 2.0, 1.5, 1.0, 0.8 | 随预测时域衰减 |
| 优化器 | AdamW，lr 3e-4，wd 1e-4 | head 训练稳定 |
| 调度 | Cosine，300 epochs | 足够收敛，单卡几分钟 |
| batch | 64（DINO 特征）/ 32（TinyCNN） | 特征训练可放大 |
| 增强 | 特征高斯噪声 0.02 + dropout 5%；图像水平镜像（同时翻转 y） | 抑制过拟合 |
| 模型选择 | 验证圈 ADE 最小 | 不用 loss 选模型，避免平滑项干扰 |

产物：`output/training/<backbone>_best.pt`、`_history.json`、`_split.json`。

---

## 4. 实验矩阵

| ID | 目的 | 设置 | 通过标准 |
|---|---|---|---|
| **E0** | 链路冒烟 | TinyCNN，单圈旧数据，5 epochs，CPU | loss 下降、预测/闭环脚本可跑 |
| **E1** | 主基线 | DINOv3，训练 1–4 圈，验证第 5 圈（72 km/h） | ADE < 3.0 m，FDE < 5.0 m |
| **E2** | 速度泛化 | 5 折 leave-one-lap-out | 五折 ADE 均值 < 3.5 m，最差折 < 5 m |
| **E3** | 车速输入必要性 | E1 + `--zero-prop` | 带车速的 ADE 至少低 30%，或闭环明显更稳 |
| **E4** | 时域 3 s vs 5 s | 重建 5 s 数据集后同样训练 | 5 s 的近 6 点精度不显著更好，且远端误差更大 |
| **E5** | 图像密度 | `--train-stride 2`（相当于 1 Hz 训练） | 与 2 Hz 的 ADE 差距 < 20% 则可用 1 Hz 再省一半存储 |
| **E6** | 闭环速度扫描 | E1 模型在 40/48/56/64/72 km/h 各跑 1 圈 | 每档完圈、无碰撞、平均横向 < 1.5 m |
| **E7** | 鲁棒性（可选） | 图像加亮度/噪声扰动后离线预测 | ADE 退化 < 30% |

> 这组实验在公开文献里的位置、与 UniAD/VAD/TransFuser 的规格对照，见
> [PROJECT_LOCATION.md](PROJECT_LOCATION.md)。E1–E7 属于「不动 baseline-v2」的纯训练/评测实验，
> 任何需要改数据的想法（加历史帧、加导航命令、DAgger、多地图）都应另立 baseline-v3，不在此列。

### 4.1 E1 主实验命令

```bash
cd waypoint_pipeline
bash ../components/dinov3/download_weights.sh          # 首次
python training/extract_features.py \
    --npz output/dataset_waypoints_5laps.npz --device cuda
python training/train.py --backbone dinov3 --device cuda \
    --val-laps 5 --epochs 300 --batch-size 64
python training/predict.py --ckpt output/training/dinov3_best.pt --device cuda
```

### 4.2 E2 五折命令

```bash
for k in 1 2 3 4 5; do
  python training/train.py --backbone dinov3 --device cuda \
      --val-laps $k --epochs 300 --out-dir output/training/fold$k
  python training/predict.py --ckpt output/training/fold$k/dinov3_best.pt \
      --out-dir output/training/fold$k/predict --device cuda
done
```

汇总各折 `pred_metrics.json` 的 `ade_m`、`fde_m`，报告均值/标准差。

### 4.3 E3 车速消融

```bash
python training/train.py --backbone dinov3 --device cuda --val-laps 5 \
    --zero-prop --out-dir output/training/no_prop
python training/predict.py --ckpt output/training/no_prop/dinov3_best.pt \
    --out-dir output/training/no_prop/predict --device cuda
```

同一验证圈比较 `ade_m`/`fde_m`。预期：无车速输入时，72 km/h 验证圈的 FDE 显著变大，
且闭环在 40/72 km/h 两端速度下偏差方向相反。

### 4.4 E4 时域消融（需要重建数据）

临时把 `config.WP_HORIZON_S` 改为 5.0、`WP_COUNT` 自动变 10，重建 npz 与特征缓存，
训练同样结构（`--loss-weights` 传 10 个值），最后把前 6 个点与 3 s 模型对齐比较。
注意：5 s 模型的远端点误差更大，不能只看总 ADE，必须分点比较。

### 4.5 E5 图像密度

```bash
python training/train.py --backbone dinov3 --device cuda --val-laps 5 \
    --train-stride 2 --out-dir output/training/stride2
```

`--train-stride` 只抽稀训练帧，验证帧保持不变，保证可比。
注意：正式数据已是 2 Hz（0.5 s 一帧），stride 2 相当于 1 Hz。

### 4.6 E6 闭环速度扫描

```bash
for v in 40 48 56 64 72; do
  python training/closed_loop.py --ckpt output/training/dinov3_best.pt \
      --speed $v --laps 1 --device cuda \
      --out-dir output/training/closed_loop/speed$v
done
```

---

## 5. Prediction 协议（离线）

- 只用验证圈，逐帧预测，指标：每点 RMSE、ADE、FDE、按圈/按速度分桶。
- 必须报告 6 个点的分点误差曲线，而不是只报 ADE；远端点的退化是时域选择的依据。
- 可视化：`pred_vs_gt.png`（6 个样本的 ego 系折线）、`per_point_rmse.png`。
- 预测脚本会输出 `pred_per_frame.csv`，便于检查最差帧（通常是急弯入口/出弯）。

**判定**：E1 的 ADE < 3 m、FDE < 5 m 视为离线可用；若 FDE 明显大于 ADE 的 2 倍，
说明远端不可用，应检查时域/数据而非继续调 head。

---

## 6. Test 协议（闭环）

- 环境：`SimpleTrackEnv`，无交通，10 Hz 控制，相机 640×360/FOV 60°。
- 控制：模型预测 → `GeometricController`（Pure Pursuit + PID + 曲率限速）。
- 每档速度跑 1 圈；记录：完圈、平均/P95/最大横向误差、出界率、碰撞、均速、推理耗时。
- **通过标准**：完圈、0 碰撞、平均 |lat| < 1.5 m、出界率 ≤ max(1%, 专家基线+0.5%)。
- 72 km/h 是 E1 的未见速度，作为泛化测试；若失败，先看是“预测偏短”还是“控制器饱和”
  （`closed_loop_log.csv` 的 `wp0_x` 与 `lat_err_m` 可区分）。
- CPU 推理慢时用 `--infer-every 2/3`，中间用位姿变换复用 waypoints；报告 `infer_ms`。

---

## 7. 产物与命名

```
output/training/
├── dinov3_best.pt                 # E1 模型
├── dinov3_history.json
├── dinov3_split.json
├── predict_dinov3/
│   ├── pred_metrics.json
│   ├── pred_arrays.npz
│   ├── pred_per_frame.csv
│   ├── pred_vs_gt.png
│   └── per_point_rmse.png
├── fold1..5/                      # E2
├── no_prop/                       # E3
├── stride5/                       # E5
└── closed_loop/speed40..72/       # E6
```

所有 `output/` 均在 `.gitignore` 中，只提交代码与文档。

**每个实验必须产出离线诊断图**（阶段 8）：

```bash
python training/report.py --version auto --title "E1 DINOv3 全量 1-4 圈" \
    --ckpt output/training/dinov3_best.pt
```

- 产物：`reports/<vN>/summary.png` + `summary_metrics.json`，并同步 `reports/latest/`；
- 任何训练代码 / 超参 / 策略 / 评测改动都要新建 `vN+1`，
  并把「图 + 读图点评 + 下一步」追加到 [ITERATION_LOG.md](ITERATION_LOG.md)；
- 对外只交付 `reports/latest/summary.png`；历史版本仅内部复盘用；
- 诊断图只读，不允许因此回改 baseline-v2 数据。

---

## 8. 计算预算（单卡，量级估计）

| 步骤 | 数据量 | 估计 |
|---|---|---|
| DINOv3 特征提取 | 7,274 帧 | GPU 1–2 min（CPU 12–25 min） |
| head 训练 300 epochs | 6,178×384 | < 2 min |
| TinyCNN 冒烟 | 数百帧 | CPU 1–2 min |
| 离线预测 | 1,096 帧（验证圈带图帧） | GPU < 1 min |
| 闭环一圈 | 40 km/h ~16 min 仿真 | 实时性取决于渲染与推理；GPU 约 1–3 min |

---

## 9. 风险与对策

| 风险 | 现象 | 对策 |
|---|---|---|
| 单帧无法判断速度 | 不同速度下 waypoint 长度系统性偏短/偏长 | E3 消融；正式模型必须带 speed |
| 相邻帧泄漏 | 验证 ADE 异常低、闭环却差 | 按圈划分；报告 split |
| 图像与训练分布不一致 | 闭环图像是实时渲染，训练图像来自录制 | 同一 `capture_rgb` 路径；同 FOV/分辨率 |
| 72 km/h 未见速度 | 闭环外扩/转向饱和 | 先看 `wp0_x` 是否 ≈ 10 m；必要时降速或加入 72 训练 |
| DINOv3 权重/版本 | `AutoModel` 加载失败 | 固定 `transformers>=4.56`，本地权重随仓库外置 |
| CPU 推理达不到 10 Hz | 控制周期抖动 | `--infer-every` 复用 + 位姿变换；正式测试用 GPU |

---

## 10. 执行顺序（建议）

1. **E0**：本机 CPU 跑 `selftest.py` + TinyCNN 5 epochs，确认代码链路。
2. **E1**：GPU 提特征 + 训练 + 离线预测，先看分点 RMSE。
3. **E6**：E1 模型闭环跑 56 km/h，确认控制器能兜住预测误差。
4. **E2/E3**：补速度泛化与消融，决定是否需要在 72 km/h 上补数据。
5. **E5/E4**：按存储/时域决策需要再做。
6. 最后把 `pred_metrics.json`、`closed_loop_metrics.json` 汇总进交付报告。

> 每一步（E0–E6）结束后都必须跑 `training/report.py` 生成该版本的 `summary.png`，
> 并在 [ITERATION_LOG.md](ITERATION_LOG.md) 追加一条点评；没有图的实验视为未完成。

---

## 11. 一页命令速查

```bash
# 本机（无 GPU）自检
cd waypoint_pipeline
.venv-train/bin/python training/selftest.py
.venv-train/bin/python training/train.py --backbone tinycnn \
    --npz output/dataset_waypoints.npz --epochs 5 --limit-train 160 --device cpu

# 训练机
python training/extract_features.py --npz output/dataset_waypoints_5laps.npz --device cuda
python training/train.py --backbone dinov3 --device cuda --val-laps 5
python training/predict.py --ckpt output/training/dinov3_best.pt --device cuda
python training/closed_loop.py --ckpt output/training/dinov3_best.pt --speed 56 --device cuda
```

> 时间 waypoints 的定义、圈尾处理、`lap_time_s` 非严格周期等实现细节，见
> [方案.md](方案.md) §3。