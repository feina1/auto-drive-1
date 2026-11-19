# Baseline 数据与规格冻结说明

> 本文件定义训练的**唯一合法数据来源**：专家按原方案正常跑线，不做任何人为调整。
> 好与坏都如实保留，作为正式对照基线。后续任何"困难数据/扰动/增广"都必须另立
> baseline v3+，**不得回改已有版本**。

相关：总览见 [README.md](README.md)，数据生产与训练交接见 [HANDOFF.md](HANDOFF.md)，
训练实验设计见 [TRAINING_PLAN.md](TRAINING_PLAN.md)，与文献的定位见 [PROJECT_LOCATION.md](PROJECT_LOCATION.md)。

---

## 0. 先区分三个频率（最容易混）

| 常量 | 值 | 作用在哪一端 | 含义 |
|---|---|---|---|
| `RECORD_HZ` | **10 Hz** | 遥测/定位（输入侧的地面真值） | 位姿、速度、转向等每 0.1 s 记一条，**始终全量** |
| `WP_FREQ_HZ` | **2 Hz** | **标签**（输出侧） | 未来 3 s 内每 0.5 s 一个 waypoint，共 **6 点** |
| `IMAGE_INTERVAL_S` | **0.5 s（2 Hz）** | **图像**（输入侧） | 每 0.5 s 拍一张图，决定**训练样本的时间密度** |

**两个 "2 Hz" 不是同一件事**：

- `WP_FREQ_HZ = 2` 回答"模型要预测多密"——从当前时刻起 t+0.5 … t+3.0，一共 6 个点；
- `IMAGE_INTERVAL_S = 0.5` 回答"模型多久看一次画面"——它只决定有多少训练样本；
- 遥测/定位不受图像频率影响，永远是 10 Hz。

两者数值相同是一个**方便的性质**：图像帧与标签网格严格重合（见 §5 的对齐校验），
但改变其中一个不会自动改变另一个。

---

## 1. 两条基线的唯一差异

| 项 | baseline-v1（归档） | baseline-v2（当前有效） |
|---|---|---|
| 图像采集频率 | **5 Hz**（0.2 s/帧） | **2 Hz**（0.5 s/帧） |
| 图片数量 | 18,185 | **7,274** |
| 图片体积 | 669.6 MB | **267.8 MB** |
| 带图训练/验证帧 | 15,391 / 2,794 | **6,178 / 1,096** |
| 位置 | `output/archive/baseline-v1-hz5/` | `output/`（根目录） |
| 状态 | 只读，可回查 | 当前唯一合法数据 |
| 其余全部相同 | 专家、圈速、场景、随机种子、标签规格、遥测频率、出界/碰撞 | 同左 |

**为什么另立 v2 而不是原地改 v1**：图像采集频率属于 [§8 变更政策](#8-变更政策) 列出的
冻结规格；v1 的 sha256 清单已经记录在案，原地修改会让历史结果无法追溯。

**为什么不能从 v1 抽稀得到 v2**：v1 的图在 t = 0.2, 0.4, 0.6, 0.8, 1.0 …，
2 Hz 需要 t = 0.5, 1.0, 1.5 …；0.5 s 处没有图。5 与 2 的公倍数是 1 s，
抽稀只能得到 1 Hz 或错位的 2 Hz。**必须重新录制。**

### 1.1 v1 与 v2 的等价性验证（已实测，2026-11-18）

为了确认 v2 不是“另一条轨迹”、只是采样更稀，做了两项对比：

| 对比项 | 方法 | 结果 |
|---|---|---|
| 相同时刻的图像 | 取 frame_0000010（t=1.0 s）、frame_0000020（t=2.0 s）对比 sha256 | **字节完全一致** |
| 全部遥测与标签 | 两个 npz 逐字段求最大偏差：x / y / heading / speed / steering / lap / t / wp_ego | **全部为 0**（逐位相同） |

结论：v1 与 v2 是**同一条专家轨迹**，唯一差异是图像采样密度（5 Hz → 2 Hz）。
因此 v1 上的结论可以沿“图像更密/更稀”这一个维度与 v2 直接对照，
但**训练样本数不同**（18,185 vs 7,274），模型指标不能直接比较。

---

## 2. Baseline v2 是什么（当前有效）

| 项 | 定义 |
|---|---|
| 数据生成 | 既有专家 `PathFollower`（Pure Pursuit + PID）沿参考线（内侧车道中心线）正常行驶 |
| 扰动 | **无**：无噪声、无侧风、无随机打方向、无速度抖动注入 |
| 障碍/交通 | **无**：`traffic_density=0.0`，无 NPC、无锥桶 |
| 随机性 | 固定 `seed=0`；赛道由固定 `TRACK_SEGMENTS` 生成；`random_lane_width/num=False` |
| 圈数 × 速度 | 5 圈：40 / 48 / 56 / 64 / 72 km/h（唯一变量，按需求逐圈切换） |
| 遥测 | **10 Hz 全量**：位姿、速度、velocity、横摆角速度、加速度、转向/油门/刹车、出界/碰撞、圈号/圈内时间/目标速度 |
| 图像 | 640×360，FOV 60°，JPEG q90，**2 Hz（0.5 s/帧，7,274 张，267.8 MB）** |
| 标签 | 未来 3 s @ 2 Hz 共 6 点，ego 系（x 前 y 左），单位 m |
| 圈尾处理 | 多录 6 s 真实未来；仅最后 30 帧按末速度直线延拓（圈尾为直道），`extrapolated_points=0` |
| 划分 | 训练 1–4 圈（30,894 帧遥测，其中 **6,178 帧带图**）/ 验证第 5 圈（5,477 帧，**1,096 帧带图**，72 km/h 未见速度） |

**明确不做的事**：不剔除出界帧、不修剪坏样本、不按 loss 筛帧、不改圈速、不重采样、
不换参考线、不调专家参数。专家自身出界 447 帧（1.23%）**原样保留**——这是基线的一部分。

---

## 3. 冻结的规格（改任一值 = 新 baseline）

来源：`config.py`（`baseline_check.py` 会逐项校验）。

| 常量 | 值 |
|---|---|
| `BASELINE_ID` | `baseline-v2` |
| `WP_HORIZON_S / WP_FREQ_HZ / WP_COUNT` | 3.0 s / 2.0 Hz / 6 |
| `SIM_DT / RECORD_HZ` | 0.1 s / 10 Hz |
| `N_LAPS / LAP_SPEEDS_KMH` | 5 / (40, 48, 56, 64, 72) |
| `TAIL_EXTRA_S` | 6.0 s |
| `IMAGE_INTERVAL_S` | **0.5 s（2 Hz）** |
| `IMAGE_WIDTH×HEIGHT, FOV, FORMAT, QUALITY` | 640×360, 60°, jpg, 90 |
| 训练归一化 | `speed/72`、`steering/0.6`（`training/common.py`） |
| 参考线文件 | `simple_track_reference_clean.json` |

---

## 4. 归档产物与校验值（v2）

目录：`waypoint_pipeline/output/`（不入 git，靠本节的校验值识别）。

| 文件 | 大小 | SHA-256 |
|---|---|---|
| `trajectory_5laps.json` | 18,408,875 B | `63888b09d80bf201aa86a1faeff59eedba05886efc8d655ec4dbfa9e04d7425d` |
| `dataset_waypoints_5laps.npz` | 3,145,910 B | `a831ff58383c2c4d3066987521aac04b49a71156b56f5c93212ae8f1d90fed17` |
| `dataset_waypoints_5laps_meta.json` | 2,769 B | `7d23d08040b084acf65434ab04f794af079ec804f1b648ce73d00ba3dd82894c` |
| `images_trajectory_5laps/` | 267,841,207 B / 7,274 files | 目录摘要 `9f7b1a722779fe741ad5c171fcda9f0b3f30b977b9830220c25604f846293b89` |

> 图片目录摘要 = 对「排序后的 `文件名\t字节数`」做 SHA-256，不逐张哈希内容，
> 用于快速判断目录是否被替换/增删（秒级完成）。
>
> 清单文件：`output/baseline_manifest.json`（由 `baseline_check.py --write-manifest` 生成，
> 含 git commit、环境版本、上述校验值与不变量）。

---

## 5. 归档不变量（v2 复现判定标准）

| 项 | 值 |
|---|---|
| 总帧数（遥测 10 Hz） | 36,371（3,637.1 s ≈ 1.01 h） |
| 总图片数（图像 2 Hz） | **7,274** |
| 每圈帧数 | 9,713 / 8,124 / 6,963 / 6,094 / 5,477 |
| 每圈图片数 | 1,942 / 1,625 / 1,393 / 1,218 / 1,096 |
| 每圈时长 | 971.3 / 812.4 / 696.3 / 609.4 / 541.6 s |
| 每圈 `|wp0|` 中位 | 5.556 / 6.667 / 7.778 / 8.889 / 10.000 m（= 0.5 s × 车速） |
| `|wp0|` max − median | ≤ 0.14 m |
| 间距中位相对误差 | 0.0001（阈值 0.01） |
| 碰撞帧 | 0 |
| 出界帧 | 447（1.23%，保留） |
| 训练/验证（遥测帧） | 30,894 / 5,477（第 5 圈） |
| 训练/验证（带图帧） | **6,178 / 1,096** |

**图像-标签对齐（额外验证，2 Hz 的收益）**

- 带图帧的相邻间隔严格为 **0.5000 s**（min = max = 0.5），无抖动；
- 带图帧的时间戳全部落在 t = 0.5, 1.0, 1.5 … 网格上，
  与 waypoint 标签的 0.5 s 采样网格**完全重合**；
- 含义：每个训练样本的图像时刻就是标签起点，不存在"图比标签早 0.1–0.4 s"的错位。

**复现判定规则**

- 对**归档副本**：SHA-256 必须完全一致（用于判断拷贝/传输是否损坏、是否被替换）。
  默认哈希不一致只告警；验收/交付时用 `python3 baseline_check.py --strict-hash` 强制计为失败。
- 对**重新录制/重新建集**：不要求字节一致（轨迹 JSON 含 `collect_time`/`wall_time_s`，
  npz 的 zip 时间戳也随每次写盘变化），但必须满足上表不变量；
  用 `python3 baseline_check.py --allow-drift` 判定（计数 ±2%、wp0 ±0.15 m）。
- 训练是否可比，另看 `TRAINING_PLAN.md` 的划分与随机种子；**数据侧不允许有任何差异**。

---

## 6. 复现命令（v2，原样执行，勿改参数）

```bash
cd waypoint_pipeline

# 1) 录制（专家正常跑线，5 圈变速，10 Hz 遥测 + 2 Hz 图像）
python3 collect_trajectory.py --laps 5 --speeds 40,48,56,64,72 \
    --images --out output/trajectory_5laps.json

# 2) 建集（3 s @ 2 Hz 6 点）
python3 make_waypoints.py --traj output/trajectory_5laps.json \
    --out output/dataset_waypoints_5laps

# 3) 冻结校验（不变量 + sha256）
python3 baseline_check.py --write-manifest
```

环境记录（本次归档）：macOS 26.5.2 arm64 / Python 3.9.6 / numpy 2.0.2 /
Panda3D 1.10.16 / MetaDrive 0.4.3 / 代码 commit `38d8286`。
录制耗时 245.3 s（v1 的 5 Hz 为 457.8 s）。

> 注意：录制时 `collect_trajectory.py` 由 `/usr/bin/python3` 执行（MetaDrive 装在该解释器）；
> 训练栈用 `.venv-train/bin/python`。两者共用同一套 numpy（2.0.2）。

---

## 7. Baseline v1（归档，5 Hz）

`output/archive/baseline-v1-hz5/`，**只读**，保留用于追溯。

| 项 | 值 |
|---|---|
| 图像 | 5 Hz（0.2 s/帧），18,185 张，669.6 MB |
| 校验值 | `trajectory_5laps.json` `99191225…c9e1`；`dataset_waypoints_5laps.npz` `45332a0b…814a`；`meta` `539100a4…26a3`；图片目录摘要 `281e1568…4ff6` |
| 带图帧 | 18,185（每个 10 Hz 帧的偶数帧） |

校验方式：

```bash
python3 baseline_check.py --baseline v1          # 只校验数据不变量 + 哈希, 跳过 config
python3 baseline_check.py --baseline v1 --strict-hash
```

> 因为 `config.py` 已前进到 v2，`--baseline v1` 会跳过 `config.*` 检查
> （当前 config 不该等于 v1 规格），只校验归档数据本身。
>
> **v1 的数据集与 v2 不可混用**：任何在 v1 上跑出的指标都不能称作 v2 结果，反之亦然。

---

## 8. 变更政策

1. **v1 / v2 只读**。已归档的轨迹/图片/npz/meta 不再改动；发现问题只记录，不"修数据"。
2. 任何以下改动都必须新建 `BASELINE_ID`（v3、v4…）并在本文件追加一节：
   时域/频率/点数、圈速或圈数、参考线、专家参数、**图像规格或采集频率**、采集脚本行为、
   筛选/过滤规则。
3. **困难数据、扰动、域随机化、对抗样本**：后续以独立 baseline 或独立数据集出现，
   不得混入已有版本，也不得用它们重训后仍称"v1/v2 结果"。
4. 训练超参（loss 权重、lr、epoch）不属于数据基线，记录在 `TRAINING_PLAN.md`；
   但同一实验组内不得边跑边改，改动需另开实验编号（E1a/E1b…）。
5. 交付/论文中引用数据时，写 `baseline-v2` + §5 不变量 + git commit，而不是只写"5 圈数据"。
6. 图像频率下调的直接后果要一并声明：**训练样本从 18,185 降到 7,274（-60%）**，
   这是用样本密度换存储/带宽的取舍；E5 会量化它对精度的影响。

---

## 9. 自检

```bash
python3 baseline_check.py                  # 严格校验 v2（93 项不变量；有清单时 +4 项 sha256）
python3 baseline_check.py --strict-hash    # 验收/交付: sha256 不一致也计为失败
python3 baseline_check.py --allow-drift    # 重录数据的宽松校验
python3 baseline_check.py --baseline v1    # 校验归档的 5 Hz 旧数据（只查数据）
```

通过标准：全部 PASS、退出码 0；任何 FAIL 都意味着「这不是对应版本的数据」，
先查文件是否被替换、config 是否被改动、或数据是否被重新生成过。