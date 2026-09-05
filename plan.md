# 一段式自动驾驶 Demo Plan

## 1. 目标

基于 MetaDrive 逐步完成一个最小一段式自动驾驶 Demo：

```text
Step 1：GT Trajectory
        → Controller
        → Vehicle

Plan 1.5：Record + GT Label（V0 中心线 / V1 Recovery）
        → Dataset

Step 2：Front RGB
        → Vision Encoder
        → Waypoint Head
        → Controller
        → Vehicle

Step 3：Front RGB
        → Vision Encoder
        → VLA Planner
        → Waypoints
        → Controller
        → Vehicle
```

三步逐层增加能力，前一步作为后一步的基础和 Baseline。

---

## 2. 三阶段定义


| 阶段       | 目标                        | 输入                            | 核心模型                         | 输出                    | 评价                            |
| -------- | ------------------------- | ----------------------------- | ---------------------------- | --------------------- | ----------------------------- |
| Step 1   | 验证循迹控制                    | GT Trajectory + Vehicle State | Pure Pursuit + P/PID         | Vehicle Control       | Closed-loop L2、Off-road       |
| Plan 1.5 | 录制并生成 GT 训练集              | Pose + RGB + 中心线              | GT V0 / V1 Labeler           | Ego Waypoints Dataset | 标签可视化、Dataset 可训练             |
| Step 2   | 验证最简单 Vision-to-Waypoints | Front RGB                     | DINOv2 + Waypoint Head       | Ego Waypoints         | Waypoint L2、Closed-loop L2    |
| Step 3   | 验证 VLA 是否改善规划能力           | Front RGB                     | Vision Encoder + VLA Planner | Ego Waypoints         | Waypoint L2、Closed-loop L2、泛化 |


---

# Step 1：轨迹循迹控制

## 方案

```text
GT Global Trajectory
        +
Vehicle position / heading / speed
        ↓
Pure Pursuit + P/PID Speed Control
        ↓
steering / throttle / brake
```

## 评价

车辆位置到 GT 轨迹最近点的距离：


d_t=\min_{p\in GT}position_t-p_2


统计：

- Mean L2
- Max L2
- Off-road

### 成功标准

```text
连续完成 ≥ 5 圈
无 Off-road
Mean L2 稳定、不逐圈发散
```

---

# Step 1.5：录制生成 GT 训练集

Step 1 验证闭环控制；Plan 1.5 在此基础上**离线录制 + 标注 GT Waypoints**，为 Step 2 的 Dataset / DINOv2 pipeline 提供监督信号。

## 总体流程

```text
Closed-loop Demo（drive_simple_track.py）
        ↓
每帧录制：pose / speed / front RGB /（
        ↓
GT Labeler（V0 → V1）
        ↓
Dataset：{image, ego_state, gt_waypoints, meta}
        ↓
Step 2：DINOv2 + Waypoint Head 训练
```

## 前置依赖（与 Step 1 共用）

- **Global Reference Path**：闭环赛道中心线（从 `road_network` 拼接，整圈 ~10800m）
- **每帧截取**：根据当前 pose，沿中心线取前方局部参考段，并转换到 ego 坐标系
- **录制格式**：建议 `episode/` + `frames.jsonl` 或 HDF5；每帧至少包含：
  - `t`, `x`, `y`, `heading`, `speed`
  - `front_rgb` 路径或 tensor
  - `gt_waypoints`（ego 系，N 个点）

---

## GT V0：中心线 Oracle（先跑通 pipeline）

**目标**：用最简单的规则 GT，先把 **Dataset → DINOv2 → Waypoint Head → 训练/评估** 整条链路跑通。

### 生成逻辑

```text
当前 pose (x, y, heading)
        ↓
在 GT Path（中心线）上找最近点（投影 / 弧长参数 s₀）
        ↓
沿中心线向前取未来距离：5 / 10 / 15 / … / 30 m（可配置步长）
        ↓
每个点 (xᵢ, yᵢ) 转换到 ego frame → [(x₁,y₁), …, (xₙ,yₙ)]
```

### 特点


| 项    | 说明                                       |
| ---- | ---------------------------------------- |
| 适用场景 | 车辆基本在中心线附近、正常循迹                          |
| 优点   | 实现简单、无额外 planner、标签稳定                    |
| 局限   | 偏离较大时，GT 仍是「回中心线前的中心线延伸」，不适合 recovery 监督 |


### V0 成功标准

```text
Dataset 可稳定导出（≥1 圈完整 episode）
Waypoint 可视化与 ego 前方路况一致
Step 2 训练 loss 能正常下降（不要求闭环已完美）
```

---

## GT V1：带 Recovery 监督

**目标**：在 V0 基础上，当横向偏差较大时，GT 不再是「纯中心线延伸」，而是一条**平滑回归中心线**的参考轨迹，用于 recovery 场景的 supervision。

### 分流规则

```text
CTE = 当前 pose 到中心线的横向偏差（Cross Track Error）

|CTE| < 0.3 m  →  GT V0：直接沿中心线取未来 waypoints

|CTE| ≥ 0.3 m  →  GT V1：在未来 20~40 m 内平滑回到中心线，再沿中心线延伸
```

> `0.3 m`、`20~40 m` 为初版超参，录制时可写入 `meta` 便于 ablation。

### V1 轨迹形状（Frenet 横向五次多项式）

在 Frenet 坐标系下，以弧长 `s` 为自变量，构造横向偏移 `l(s)`：


l(s):\quad l_0 \rightarrow 0


**边界条件（示意）**：


| 位置                                | 横向 `l`     | 横向导数 / 航向      |
| --------------------------------- | ---------- | -------------- |
| 起点 `s = s₀`                       | `l₀ = CTE` | 与当前车辆航向相对中心线一致 |
| 终点 `s = s₀ + L`（`L ∈ [20, 40]` m） | `l = 0`    | 航向与中心线切向一致     |


五次多项式可唯一确定 `l(s)`，满足起终点的位置、一阶（航向）约束；纵向沿 `s` 匀速或按目标速度参数化即可。

```text
当前位置 + 当前 CTE + 当前航向
        ↓
Frenet 五次多项式 l(s)，在 [s₀, s₀+L] 上 l: l₀ → 0
        ↓
转换回全局 (x, y)，再均匀采样为 waypoints
        ↓
转 ego frame → gt_waypoints
```

### V1 与 V0 的关系

```text
V0：Oracle 中心线标签     → 正常驾驶、Step 2 主监督
V1：Recovery 回归标签     → 大偏差样本、增强泛化与回线能力

Dataset 可并存：
  - gt_waypoints_v0
  - gt_waypoints_v1
  - label_type / |CTE| 写入 meta，训练时可混合或分阶段
```

### V1 成功标准

```text
|CTE| ≥ 0.3 m 的帧：可视化 GT 曲线平滑贴回中心线，无折线/跳变
Recovery 段与 V0 段在 CTE 阈值附近过渡自然
Step 2 在大偏差场景 closed-loop 优于仅 V0 训练（后续验证）
```

---

## Plan 1.5 实施顺序（建议）


| 顺序  | 任务                              | 产出                  |
| --- | ------------------------------- | ------------------- |
| 1   | 中心线 Global Path 构建 + 最近点 / 弧长查询 | `reference_path.py` |
| 2   | GT V0 labeler + 单帧可视化           | ego waypoints 图     |
| 3   | 闭环录制脚本（RGB + pose + GT）         | raw episodes        |
| 4   | Dataset 打包 + Step 2 dataloader  | train/val split     |
| 5   | GT V1 Frenet recovery labeler   | `gt_waypoints_v1`   |
| 6   | 混合数据集训练 DINOv2 baseline         | Step 2 pipeline 跑通  |


---

# Step 2：纯视觉一段式方案

## 2.1 循迹训练

相关逻辑统一放在 `model/follow_path.py`，分为三部分：

### dataset：生成循迹训练数据

- 从 `records/train` 读取数据。
- 输入只取单帧 front RGB 和自车速度。
- 监督标签为对应的 GT ego waypoints。

### model：模型设计

- 单帧图片经预训练 DINOv2 ViT-S/14 提取特征，再与自车速度拼接融合。
- 权重无需登录，首次下载到 `model/.cache/huggingface`，后续优先复用本地缓存。
- 使用简单 MLP 输出 ego 坐标系下的 waypoints：`[(x1, y1), ..., (xN, yN)]`。
- 初版冻结 DINOv2，只训练 MLP。

### 训练

- 使用预测 waypoints 与 GT waypoints 的 MSE 作为简单循迹 loss。
- 先跑通数据读取、前向预测和训练，使 loss 正常下降。

---

# Step 3：基于 VLA 的一段式方案

## 方案

Step 3 保留视觉编码过程，在 Step 2 基础上增加更强的规划 / 推理模块：

```text
Front RGB
    ↓
Vision Encoder
    ↓
Visual Tokens
    ↓
Projector / Adapter
    ↓
VLA / LLM-based Planner
    ↓
Waypoint Head
    ↓
Ego Waypoints
    ↓
同一个 Controller
```

第一版优先考虑复用 Step 2 的 DINOv2：

```text
Step 2：

RGB
 ↓
DINOv2
 ↓
MLP
 ↓
Waypoints


Step 3：

RGB
 ↓
DINOv2
 ↓
Visual Tokens
 ↓
VLA / LLM Planner
 ↓
Waypoints
```

因此 Step 3 主要验证：

> 在相同视觉信息基础上，引入 VLA 的语义理解、预训练知识和规划推理能力，能否提升实际自动驾驶效果。

当前阶段**不重点研究自然语言交互**。

如果采用现成 VLA/VLM，也可以使用模型自带的 Vision Encoder，不强制使用 DINOv2。

## 评价

与 Step 2 保持完全一致：

- Waypoint L2
- Closed-loop Trajectory L2
- Off-road
- 新赛道 / 新环境泛化性能

重点比较：


| Metric         | Step 2 Vision | Step 3 VLA |
| -------------- | ------------- | ---------- |
| Waypoint L2    |               |            |
| Closed-loop L2 |               |            |
| Off-road       |               |            |
| New Track L2   |               |            |


只有当 Step 3 在轨迹精度、闭环稳定性或泛化能力上优于 Step 2，才认为 VLA 对自动驾驶任务本身产生了实际增益。

---

## 3. 最终演进关系

```text
Step 1
GT Path
   ↓
Controller
   ↓
Vehicle

        ↓

Plan 1.5
Record RGB + Pose
   ↓
GT V0（中心线）/ GT V1（Recovery）
   ↓
Training Dataset

        ↓

Step 2
RGB → DINOv2 → Waypoint Head
                    ↓
                 Waypoints
                    ↓
                Controller

        ↓

Step 3
RGB → Vision Encoder → Visual Tokens
                           ↓
                       VLA Planner
                           ↓
                        Waypoints
                           ↓
                       Controller
```

整个项目最终要回答两个问题：

1. **纯视觉一段式方案是否能够完成稳定闭环驾驶？**
2. **在相同任务下，引入 VLA 后是否真正提升驾驶与泛化能力，而不仅仅增加语言交互能力？**
