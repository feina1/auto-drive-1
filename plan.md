# 一段式自动驾驶 Demo Plan

## 1. 目标

基于 MetaDrive 逐步完成一个最小一段式自动驾驶 Demo：

```text
Step 1：GT Trajectory
        → Controller
        → Vehicle

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

| 阶段 | 目标 | 输入 | 核心模型 | 输出 | 评价 |
|---|---|---|---|---|---|
| Step 1 | 验证循迹控制 | GT Trajectory + Vehicle State | Pure Pursuit + P/PID | Vehicle Control | Closed-loop L2、Off-road |
| Step 2 | 验证最简单 Vision-to-Waypoints | Front RGB | DINOv3 + Waypoint Head | Ego Waypoints | Waypoint L2、Closed-loop L2 |
| Step 3 | 验证 VLA 是否改善规划能力 | Front RGB | Vision Encoder + VLA Planner | Ego Waypoints | Waypoint L2、Closed-loop L2、泛化 |

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

\[
d_t=\min_{p\in GT}\|position_t-p\|_2
\]

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

# Step 2：纯视觉一段式方案

## 方案

不做传统感知，直接从前视图像预测 Waypoints：

```text
Front RGB
    ↓
DINOv3 Vision Encoder
    ↓
Visual Features / Tokens
    ↓
Simple Waypoint Head
    ↓
Ego Waypoints
    ↓
Step 1 Controller
```

第一版：

```text
DINOv3：Frozen
Waypoint Head：Trainable MLP
```

输出：

```text
[(x1,y1), ..., (xN,yN)]
```

统一采用 ego 坐标系。

## 评价

Open-loop：

```text
Predicted Waypoints vs GT Waypoints
→ Mean Waypoint L2
```

Closed-loop：

```text
Actual Vehicle Trajectory vs GT Trajectory
→ Mean L2 + Off-road
```

Step 2 是后续 VLA 的核心 Baseline。

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

第一版优先考虑复用 Step 2 的 DINOv3：

```text
Step 2：

RGB
 ↓
DINOv3
 ↓
MLP
 ↓
Waypoints


Step 3：

RGB
 ↓
DINOv3
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

如果采用现成 VLA/VLM，也可以使用模型自带的 Vision Encoder，不强制使用 DINOv3。

## 评价

与 Step 2 保持完全一致：

- Waypoint L2
- Closed-loop Trajectory L2
- Off-road
- 新赛道 / 新环境泛化性能

重点比较：

| Metric | Step 2 Vision | Step 3 VLA |
|---|---:|---:|
| Waypoint L2 | | |
| Closed-loop L2 | | |
| Off-road | | |
| New Track L2 | | |

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

Step 2
RGB → DINOv3 → Waypoint Head
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