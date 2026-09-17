# 2026-09-15 最终方案 (修正版)

> 本文档是对之前所有设计文档的修正汇总。
> 之前文档中的原始内容不删除，保留作为记录。
> 本文标注了 **[修正]** 的地方，是对之前文档中错误的纠正。

---

## 一、总体目标

用 MetaDrive 仿真环境，训练一个端到端模型：
- 输入：前视 RGB 图片
- 输出：前方局部参考路径 (waypoints)
- PID 控制器按预测的路径跟踪行驶
- 目标：完成 1 圈，与全局参考路径偏移 < 2m

---

## 二、模型架构

### **[修正]** 输出从 10 个点改为 20 个点

> 原设计 (prepare_dataset.md / Training.md): 10 个点, 1m 间隔, 前方 0~9m
>
> 修正原因: 70km/h 下 10m 只有 0.5 秒前瞻，不够。20m = 1 秒前瞻，给 PID 更多反应时间。

```
输入: 1 帧 RGB 图片 [3, 224, 224]
    ↓
DINOv3 ViT-S/16 (frozen, 21.6M params)
    ↓
CLS [384] + mean(patches) [384] → 平均 → [384]
    ↓
Waypoint Head (MLP, ~167K params):
    Linear(384 → 256) → GELU → Dropout(0.1)
    Linear(256 → 256) → GELU → Dropout(0.1)
    Linear(256 → 40)  → reshape → [20, 2]
    ↓
输出: 20 个 waypoints (ego frame, 1m 间隔, 前方 1m~20m)
```

### **[修正]** 先做单帧 baseline，再加时序

> 原设计 (20260915.md / prepare_dataset.md): 5 帧时序输入
>
> 修正原因: 5 帧特征取平均会丢失时间信息 (suggestion_AI.md 有详细分析)。
> 但更重要的是先跑通再迭代。单帧做通了再加时序。

Phase 1: 单帧 → 20 waypoints (先跑通)
Phase 2: 5 帧 → 20 waypoints (加时序)

Phase 2 的时序方案:
- 不用取平均 (会丢失方向信息)
- 用 5 帧的 CLS token 拼接 [5×384=1920] → temporal MLP → [384]
- 或者只用最新 1 帧 + 帧差特征

---

## 三、数据收集

### **[修正]** 从"扰动 1 秒→恢复"改为"随机偏移→PID 开回去"

> 原设计 (prepare_dataset.md):
> - 对车辆施加扰动让车漂移 1 秒，录 5 帧
> - 然后 PID 恢复，记录恢复路径作为 GT
>
> 修正原因 (suggestion_AI.md 有详细分析):
> 1. 训练时模型看到的是"偏移中的画面"，推理时看到的是"正常画面"→ 分布不匹配
> 2. 扰动 1 秒不给控制，大偏移时车可能已经出界
> 3. 流程复杂，调试困难

### 新方案: 随机偏移采样

```python
for i in range(2000):  # 2000 个样本
    # 1. 在参考路径上随机选一个位置
    idx = random.randint(0, len(ref_path) - 200)
    base_pos = ref_path[idx]
    base_heading = ref_heading[idx]
    
    # 2. 随机横向偏移和航向偏移
    lateral_offset = random.uniform(-3.0, 3.0)   # ±3m
    heading_offset = random.uniform(-15, 15)      # ±15°
    
    # 3. 在 MetaDrive 中把车放到偏移位置
    env = SimpleTrackEnv(config)
    env.reset()
    vehicle.set_position(base_pos + lateral * normal)
    vehicle.set_heading(base_heading + heading_offset)
    vehicle.set_velocity([target_speed, 0])
    
    # 4. 捕获当前图片 → 作为输入
    img = capture_image(env)
    
    # 5. 用 PID 从当前位置开回去，记录轨迹
    trajectory = []
    for step in range(300):  # 最多开 300 步
        control = pid.compute(vehicle, ref_path)
        env.step(control)
        trajectory.append({
            "position": vehicle.position,
            "heading": vehicle.heading,
        })
        if lateral_error < 0.2 and abs(heading_error) < 2:
            break  # 已经回到参考路径
    
    # 6. 从轨迹中采样 20 个 GT waypoints
    gt = resample_at_1m(trajectory, count=20)
    gt = world_to_ego(gt, vehicle.position, vehicle.heading)
    
    # 7. 保存 (img, gt)
    save(img, gt, meta={...})
    
    env.close()
```

### 数据分布设计

| 偏移范围 | 占比 | 训练目标 |
|---------|------|---------|
| ±0~0.5m | 20% | 正常行驶，保持稳定 |
| ±0.5~1.5m | 30% | 小幅偏移，轻微纠偏 |
| ±1.5~3.0m | 30% | 大幅偏移，明显纠偏 |
| ±3.0m+ | 20% | 极端偏移（到对向车道），大幅恢复 |

### 弯道/直道分布

在参考路径上选位置时，按曲率分类：
- 左弯区间: 选 30% 的样本
- 右弯区间: 选 30% 的样本
- 直行区间: 选 40% 的样本

每个区间内均匀随机选位置。

### **[修正]** 数据量从 1000 段改为 2000 个样本

> 原设计: 1000 段, 每段 5 帧
>
> 修正原因: Phase 1 用单帧，不需要"段"的概念。每个样本就是 1 张图 + 1 组 GT。
> 2000 个样本足够 Phase 1 跑通。Phase 2 加时序时再改成段。

---

## 四、GT 生成细节

### 从 PID 恢复轨迹到 GT waypoints

```python
def generate_gt(trajectory, start_pos, start_heading, ref_path, count=20):
    """
    从 PID 恢复轨迹中采样 GT waypoints.
    
    逻辑:
    1. 轨迹的第一个点就是车辆当前位置
    2. 沿轨迹每隔 1m 采样一个点
    3. 取前 20 个点 (前方 1m ~ 20m)
    4. 转换到 ego frame
    """
    # 重采样为等间距
    resampled = resample_equidistant(trajectory, interval=1.0)
    
    # 取前 count 个点 (跳过第 0 个, 因为那是当前位置)
    points = resampled[1:count+1]
    
    # 如果轨迹不够长, 用最后的方向线性外推
    while len(points) < count:
        last = points[-1]
        direction = heading_from_trajectory(points[-3:])
        next_pt = last + 1.0 * direction_vector(direction)
        points.append(next_pt)
    
    # 转换到 ego frame
    ego_points = world_to_ego(points, start_pos, start_heading)
    
    return ego_points  # shape: [20, 2]
```

### 质量过滤

跳过不合格的样本:
- PID 恢复失败 (300 步后横向误差仍 > 2m)
- 恢复过程中 offroad
- 恢复过程中 collision
- 轨迹长度 < 10m (太短，说明初始状态太差)

---

## 五、训练配置

### 超参数

| 参数 | 值 | 理由 |
|------|-----|------|
| Batch size | 64 | 单帧，显存够用 |
| Epochs | 300 | 2000 样本，不需要太多 |
| Optimizer | AdamW | 标准 |
| Learning rate | 3e-4 | 只训练 head |
| Weight decay | 1e-4 | 防过拟合 |
| Scheduler | CosineAnnealing | 平滑衰减 |
| Train/Val | 80%/20% | 1600 训练, 400 验证 |

### Loss

加权 L2:
```python
# 权重: 近处高, 远处低
weights = [2.5, 2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.2, 1.0, 1.0,
           0.8, 0.8, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3]

loss = weighted_mse(pred_waypoints, gt_waypoints, weights)
```

### 特征增强

- Gaussian noise (σ=0.02)
- Feature dropout (p=0.05)
- 图片增强: color jitter (±10%), random crop (±5%)

---

## 六、闭环测试 (run.py)

### Phase 1: 每帧推理

```python
while not done:
    # 每帧:
    img = capture_image(env)
    pred_wp = model.predict(img)          # → [20, 2]
    control = pid.follow_path(pred_wp)    # PID 跟踪预测路径
    env.step(control)
    
    # 记录横向误差 (与全局参考路径比较)
    lat_err = compute_lateral_error(vehicle.position, ref_path)
```

### Phase 3 (后续): 1Hz 推理

```python
frame_count = 0
latest_wp = None

while not done:
    img = capture_image(env)
    frame_count += 1
    
    # 每 10 帧 (1Hz) 推理一次
    if frame_count % 10 == 0:
        latest_wp = model.predict(img)
    
    # PID 用最新的 waypoints 跟踪
    if latest_wp is not None:
        control = pid.follow_path(latest_wp)
    else:
        control = pid.follow_path(global_ref_path)
    
    env.step(control)
```

---

## 七、可视化

### 离线可视化 (数据检查)

HTML 页面，每个样本显示:
- 左边: 输入图片
- 右边: GT 路径俯视图 (参考路径 + GT waypoints + 车辆位置)
- 支持翻页、筛选 (按偏移大小、弯道/直道)

### 闭环可视化 (测试结果)

- 实时显示: 前视图片 + 预测路径 + 横向误差
- 结束后统计: mean/max/90%ile 横向误差、完成圈数

---

## 八、实施步骤

### Step 1: 数据收集脚本 (今天)
- 编写 `train/collect_data.py`
- 随机偏移 → PID 恢复 → 保存图片 + GT
- 先跑 20 个样本验证流程
- 可视化检查 GT 质量

### Step 2: 正式收集 2000 个样本 (今天)
- 后台运行
- 预计 2000 个样本 × ~5 秒/个 = ~3 小时

### Step 3: 模型 + 训练 (今天)
- 编写 `train/model_v2.py` (单帧 → 20 waypoints)
- 编写 `train/train_v2.py`
- 训练 ~5 分钟

### Step 4: 闭环测试 (今天/明天)
- 编写 `train/run.py`
- 每帧推理，PID 跟踪
- 目标: 1 圈, mean lat_err < 2m

### Step 5: 迭代 (后续)
- 如果 baseline 达标: 加时序 (5 帧)、降推理频率 (1Hz)
- 如果不达标: 增加数据量、调参、分析失败案例

---

## 九、修正记录

| # | 原文档 | 原内容 | 修正为 | 原因 |
|---|--------|--------|--------|------|
| 1 | prepare_dataset.md | 10 个 waypoints, 1m 间隔 | 20 个 waypoints, 1m 间隔 | 10m 前瞻不够 (70km/h 下 0.5s) |
| 2 | 20260915.md | 5 帧时序输入 | 先单帧, 后加时序 | 先跑通再迭代; 取平均丢失时间信息 |
| 3 | prepare_dataset.md | 扰动 1 秒→录 5 帧→PID 恢复 | 随机偏移→PID 开回去→记录路径 | 分布不匹配; 流程复杂; 大偏移车会飞 |
| 4 | prepare_dataset.md | 1000 段, 每段 5 帧 | 2000 个样本, 每个 1 帧 | Phase 1 单帧不需要段 |
| 5 | prepare_dataset.md | 4 种场景各 250 段 | 按偏移大小 + 弯道/直道分布 | 更自然, 不需要人工分类场景 |
| 6 | Training.md | 5 帧特征取平均 | 单帧 CLS+patches 平均 | 简化; 取平均在时序下有问题 |
| 7 | 20260915.md | 推理频率 1Hz | Phase 1 每帧推理, Phase 3 再 1Hz | 先验证模型能力, 再优化频率 |
| 8 | solution.md | image_on_cuda, multi_thread_render | 暂不需要 | Phase 1 先不管渲染优化 |

---

## 十、文件结构

```
train/
├── dinov3-weights/vits16/        # DINOv3 预训练权重 (已有)
├── data/
│   ├── images/                    # 收集的输入图片
│   │   ├── sample_0000.jpg
│   │   └── ...
│   ├── gt/                        # GT waypoints
│   │   ├── sample_0000.json
│   │   └── ...
│   └── manifest.json              # 所有样本的元信息
├── collect_data.py                # 数据收集
├── model_v2.py                    # 模型 (单帧 → 20 waypoints)
├── train_v2.py                    # 训练脚本
├── run.py                         # 闭环测试
├── visualize_data.py              # 数据可视化
└── checkpoints/
    └── best_model.pt
```

### 之前的文件 (不再使用, 保留备查)
```
train/
├── dataset.py                     # 旧: 5m 间隔 6 个点
├── dataset_fast.py                # 旧: 预提取特征
├── model.py                       # 旧: 单帧 → 6 个点
├── train.py                       # 旧: 训练脚本
├── test_closed_loop.py            # 旧: 闭环测试
├── record_perturbed.py            # 旧: 扰动录制
```
