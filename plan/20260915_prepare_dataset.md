# 数据集准备设计

## 一、目标

准备 1000 段训练数据。每段数据教会模型：看到当前 5 帧画面，输出接下来应该走的 10 米局部参考路径。

4 种场景各 250 段：
1. 左弯 + 偏左 → 纠偏回右
2. 右弯 + 偏右 → 纠偏回左
3. 右弯 + 偏左 → 纠偏回右
4. 左弯 + 偏右 → 纠偏回左

## 二、参考路径分析

文件: `simple_track_reference_clean.json`
- 总长度: 26185m (约 26km)
- 总点数: 26266 个
- 点间距: 约 1m

弯道检测:
- 对每个点计算前后各 10 个点的曲率 (cross product)
- 曲率 > 0.05: 左弯
- 曲率 < -0.05: 右弯
- 绝对值 <= 0.05: 直行

根据曲率将路径上的点分类为 left_turn / right_turn / straight。
录制时从对应分类的区间中随机选取位置。

## 三、单段数据结构

每段 (segment) 包含:

```
segment = {
    "segment_id": 0~999,
    "scenario_type": "left_turn_left_drift" | "right_turn_right_drift" | 
                      "right_turn_left_drift" | "left_turn_right_drift",
    "ref_path_index": 在参考路径上的起始位置,
    
    # 扰动阶段 (1 秒, 5 帧)
    "disturbance": {
        "direction": "left" | "right",
        "magnitude": 横向力的大小,
        "duration_s": 1.0,
    },
    
    # 输入: 5 帧图片 (5Hz, 1 秒内)
    "input_frames": [frame_0.jpg, frame_1.jpg, ..., frame_4.jpg],
    
    # 输出 GT: 10 个 waypoints (ego frame, 1m 间隔, 前方 0~9m)
    "gt_waypoints": [[x0,y0], [x1,y1], ..., [x9,y9]],
    
    # 恢复轨迹元信息
    "recovery": {
        "start_position": [x, y],
        "start_heading": heading,
        "lateral_offset_at_start": 偏移量,
        "recovery_distance_m": 恢复用了多少米,
        "max_lateral_error": 最大横向误差,
    }
}
```

## 四、数据生成流程

对每段执行以下步骤:

### Step 1: 选择位置和场景

```python
# 从参考路径上随机选一个点
idx = random.choice(left_turn_indices)  # 或 right_turn_indices
position = ref_path[idx]
heading = ref_path_heading[idx]

# 根据场景类型决定扰动方向
if scenario == "left_turn_left_drift":
    drift_direction = "left"
elif scenario == "left_turn_right_drift":
    drift_direction = "right"
# ...
```

### Step 2: MetaDrive 初始化

```python
# 创建环境
env = SimpleTrackEnv(config)
env.reset()

# 将车辆放到选定的位置
vehicle.set_position(position)
vehicle.set_heading_theta(heading)
vehicle.set_velocity([target_speed, 0])
```

### Step 3: 扰动阶段 (1 秒, 5 帧)

```python
# 施加持续横向力, 让车往扰动方向漂移
# 录制 5 帧 (5Hz, 每 0.2 秒一帧)
for step in range(5):
    # 施加横向力 (通过 set_velocity 叠加横向分量)
    apply_lateral_force(vehicle, drift_direction, magnitude)
    env.step([0, 0])  # 不给控制, 让车自然漂移
    capture_frame()   # 保存图片
    wait(0.2)         # 等 0.2 秒 (5Hz)
```

扰动强度分级:
- 小: 横向偏移 0.2~0.5m
- 中: 横向偏移 0.5~1.2m
- 大: 横向偏移 1.2~2.5m
- 超大: 横向偏移 2.5~4.0m

各占 30%, 30%, 25%, 15%。

### Step 4: PID 恢复阶段

```python
# 停止扰动, 让 PID 接管
# PID 使用全局参考路径做跟踪
follower = PathFollower(ref_path)

# 记录恢复轨迹 (密集采样)
recovery_trajectory = []
while not recovered:
    control = follower.compute_control(vehicle)
    env.step(control)
    recovery_trajectory.append(vehicle.position)
    # 终止条件: 横向误差 < 0.3m 且持续 2 秒
```

### Step 5: 生成 GT

```python
# 从恢复轨迹的第一个点开始
# 沿恢复轨迹每隔 1m 采样, 取前 10m
raw_points = resample_at_1m(recovery_trajectory)
gt_waypoints = raw_points[:10]  # 10 个点

# 转换到 ego frame
ego_waypoints = world_to_ego(gt_waypoints, vehicle_position, vehicle_heading)
```

### Step 6: 质量检查

跳过不合格段:
- 恢复过程中 offroad → 丢弃
- 恢复过程中 collision → 丢弃
- 恢复失败 (50m 内没回到参考路径) → 丢弃
- 初始横向误差 < 0.1m (扰动没生效) → 丢弃

## 五、录制参数

| 参数 | 值 |
|------|-----|
| 总段数 | 1000 |
| 每场景 | 250 段 |
| 扰动频率 | 5Hz (5 帧/段) |
| 扰动时长 | 1 秒 |
| GT 点数 | 10 (1m 间隔, 0~9m) |
| 目标速度 | 70 km/h |
| 扰动强度分布 | 30% 小, 30% 中, 25% 大, 15% 超大 |

## 六、可视化设计

### 格式
HTML 网页, 参考 `generate_replay.py` 的 HTML 生成方式。

### 布局
```
+------------------------------------------+
|  Segment #123 / 1000                     |
|  场景: 左弯+偏左  |  扰动: 0.8m          |
+------------------------------------------+
|                    |                      |
|   5 帧输入图片     |    GT 路径图         |
|   (2x3 grid)      |    (俯视图)          |
|   frame 0 ~ 4     |    · 参考路径         |
|                    |    · 恢复轨迹         |
|                    |    · 10 个 GT 点      |
|                    |    · 车辆起始位置     |
|                    |                      |
+------------------------------------------+
|  < 上一段  |  下一段 >  |  跳转: [___]   |
|  筛选: [全部] [左弯] [右弯]               |
|  翻页: [1-100] [101-200] ... [901-1000]  |
+------------------------------------------+
```

### 5 帧图片展示
- 2 行 x 3 列 grid (最后一格空白或显示文字信息)
- 每帧下方标注时间: t=0.0s, t=0.2s, ..., t=0.8s

### GT 路径图
- 俯视图 (bird's eye view)
- 灰色线: 全局参考路径 (背景)
- 蓝色线: 恢复轨迹 (PID 实际走的)
- 红色点: 10 个 GT waypoints
- 绿色三角: 车辆起始位置和朝向
- 黄色虚线: 扰动方向

### 导航功能
- 左右箭头: 上一段/下一段
- 下拉列表: 直接跳转到指定段号
- 翻页按钮: 每页 100 段
- 筛选按钮: 全部 / 左弯左偏 / 左弯右偏 / 右弯左偏 / 右弯右偏

### 批量管理
- 所有段存在一个 JSON 文件中: `segments_manifest.json`
- 图片存在 `segments/images/` 目录下
- 可视化 HTML 可以从 manifest 重新生成
- 支持增量录制 (录了一半可以接着录)

## 七、文件结构

```
train/
├── data/
│   ├── segments/
│   │   ├── manifest.json          # 所有段的元信息
│   │   ├── images/
│   │   │   ├── seg_0000_f0.jpg    # 第 0 段 frame 0
│   │   │   ├── seg_0000_f1.jpg
│   │   │   └── ...
│   │   └── gt/
│   │       ├── seg_0000_gt.json   # 第 0 段 GT waypoints
│   │       └── ...
│   └── visualization/
│       ├── index.html              # 可视化主页
│       └── data.js                 # 数据 (从 manifest 生成)
├── prepare_dataset.py              # 数据收集主脚本
├── visualize_segments.py           # 可视化生成脚本
└── analyze_path.py                 # 参考路径分析
```

## 八、实施步骤

1. 完善 `analyze_path.py` — 分析参考路径, 输出弯道/直行区间
2. 编写 `prepare_dataset.py` — 批量生成 1000 段数据
3. 编写 `visualize_segments.py` — 生成 HTML 可视化
4. 先录 10 段验证流程正确
5. 查看可视化确认数据质量
6. 正式录制 1000 段
7. 最终可视化检查
