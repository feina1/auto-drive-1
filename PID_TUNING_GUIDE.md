# PID 控制器调试原理与实战指南

## 一、为什么需要 PID？

### 自动驾驶的两个核心问题：

1. **纵向控制**：车速应该保持在目标值（如 30km/h）
2. **横向控制**：车辆应该沿着参考路径行驶，不偏离

PID 就是解决这两个问题的经典方法！

---

## 二、PID 的基本原理

### PID 公式

```
output = Kp × error + Ki × integral + Kd × derivative
```

#### 三个项的含义：

| 项 | 含义 | 作用 | 类比 |
|-----|------|------|------|
| **P (比例)** | 当前误差×增益 | 反应越快，控制越强 | 看到前面有障碍就刹车 |
| **I (积分)** | 历史误差累积 | 消除稳态误差 | 一直偏右就慢慢向左修正 |
| **D (微分)** | 误差变化率 | 抑制震荡 | 速度快时就轻踩刹车 |

---

## 三、纵向 PID 控制器（控速）

### 目标：保持目标速度

```python
speed_error = target_speed - current_speed
throttle = KP_SPEED × error + KI_SPEED × integral
```

### 参数含义：

| 参数 | 典型值 | 含义 | 影响 |
|------|--------|------|------|
| **KP_SPEED** | 0.1-0.3 | 速度 P 增益 | 越大油门响应越快 |
| **KI_SPEED** | 0.01-0.05 | 速度 I 增益 | 防止长期偏低速 |
| **INTEGRAL_MAX** | 3.0-5.0 | 积分限幅 | 防止积分饱和 |

### 为什么要这么设置？

#### KP_SPEED 的分析：

- **太小** (<0.1)：
  - 问题：加速慢、爬坡时速度掉太多
  - 原因：油门给得太保守
  
- **适中** (0.1-0.3)：
  - ✅ 平衡性好
  - 油门能较快响应速度偏差
  
- **太大** (>0.5)：
  - 问题：油门会疯狂波动（全开→全关）
  - 原因：一小点速度偏差就被放大很多倍

#### 计算示例：

假设：
- 目标速度 = 30 km/h = 8.33 m/s
- 实际速度 = 7.5 m/s
- KP = 0.15

```
speed_error = 8.33 - 7.5 = 0.83 m/s

throttle_cmd = 0.15 × 0.83 = 0.124 (约 12% 油门)
```

如果实际速度更低 → 油门更大，合理！

### 调试步骤：

```python
# Step 1: 先设 KI=0, KP 从小到大测试
KP_SPEED = 0.05   # 初始
while 不稳定:
    KP_SPEED += 0.05   # 每次增加一点

# Step 2: 如果始终有速度偏差，再加 KI
if speed 始终低于目标:
    KI_SPEED = 0.02     # 小值起步
    
# Step 3: 如果速度慢但震荡，加 KD
if 速度波动大:
    KD_SPEED = 0.01     # 阻尼效果
```

---

## 四、横向 PID 控制器（控方向）

### 目标：保持车辆在参考路径中心线

```python
angle_error = target_heading - vehicle_heading
steering = KP_TURN × angle_error + KD_TURN × derivative
```

### Pure Pursuit 的作用：

在前方找一个参考点，计算到该点的方向应该是多少。

```
target_heading = atan2(ref_y - prev_y, ref_x - prev_x)
angle_error = target_heading - vehicle_heading
```

### 参数含义：

| 参数 | 典型值 | 含义 | 影响 |
|------|--------|------|------|
| **KP_TURN** | 0.5-1.2 | 转向 P 增益 | 越大转向越灵敏 |
| **KD_TURN** | 0.0-0.3 | 转向 D 增益 | 抑制方向盘震荡 |
| **MAX_STEER** | 0.3-0.8 rad | 最大转向角 | 约 17°~45° |
| **LOOK_DIST_BASE** | 1.5-3.0m | 基础前瞻距离 | 越远越稳定但追踪差 |
| **LOOK_SCALE** | 0.1-0.2 | 前瞻随速系数 | L = base + scale × v |

### 为什么要这么设置？

#### KP_TURN 的原理：

Pure Pursuit 本质是圆弧拟合：

```
curvature ≈ 2 × sin(angle_error) / lookahead_distance
steering ≈ curvature × wheelbase² / 2
steering_cmd = KP_TURN × steering_theoretical
```

- **KP_TURN = 0.6**：中等灵敏度
  - 低速时可以跟上弯道
  - 高速时不会过度转向
  
- **KP_TURN = 0.2**（太小）：
  - 弯道跟不住，轨迹偏离
  - 像"懒车"，转弯不积极
  
- **KP_TURN = 1.5**（太大）：
  - 稍微有点角度偏差就猛打方向
  - 容易左右震荡，乘客晕车

#### LOOK_DIST 的原理：

```python
lookahead_dist = LOOK_DIST_BASE + LOOK_SCALE × speed_ms
```

**动态调整的原因：**

- **低速时** (v < 5m/s): 
  - Lookahead = 1.5m (短)
  - 好处：可以跟踪急弯
  - 原因：车还没惯性，马上可以转过来
  
- **高速时** (v > 10m/s):
  - Lookahead = 1.5 + 0.15×10 = 3.0m (长)
  - 好处：提前预判，避免急转
  - 原因：速度快时急转会翻车

#### 曲率半径与转向的关系：

纯追踪算法推导：

```
R = L² / (4 × dy_local)     (R: 曲率半径，L: 前瞻距离)
δ = arctan(L_wb × curvature / 2)  (δ: 转向角)
```

所以：
- 前瞻距离越长 → R 越大 → 转向角越小
- 角度偏差越大 → curvature 越大 → 转向角越大

### 调试步骤：

```python
# Step 1: 先设低速测试
TARGET_SPEED = 20.0         # 很慢

# Step 2: 设置中等前瞻距离
LOOK_DIST_BASE = 2.0        # 2 米起步
LOOK_SCALE = 0.12           # 每 m/s 加 0.12 米

# Step 3: 从小 KP 开始，逐步增加
KP_TURN = 0.5               # 初始
KD_TURN = 0.1               # 一点阻尼

while True:
    运行一圈
    if 横向偏差 > 1m:       # 没跟上
        KP_TURN += 0.1      # 增大灵敏度
    elif 左右震荡明显:       # 太敏感
        KP_TURN -= 0.1      # 减小
        KD_TURN += 0.05     # 增加阻尼
    elif 偏差 < 0.1m:       # 已经很准了
        break
```

---

## 五、完整参数表

### 保守配置（新手推荐）

适合调试阶段，安全优先：

```python
class C_CONSERVATIVE:
    TARGET_SPEED = 20.0          # km/h
    MAX_STEER = 0.5              # 约 29 度
    
    # 纵向 PID
    KP_SPEED = 0.1
    KI_SPEED = 0.02
    INTEGRAL_MAX = 3.0
    
    # 横向 PID
    KP_TURN = 0.5                # 温和
    KD_TURN = 0.15               # 较多阻尼
    LOOK_DIST_BASE = 1.5         # 较短前瞻
    LOOK_SCALE = 0.1             # 随速增长慢
```

### 激进配置（稳定后）

追求更精准的控制：

```python
class C_AGGRESSIVE:
    TARGET_SPEED = 40.0          # km/h
    MAX_STEER = 0.8              # 约 46 度
    
    # 纵向 PID
    KP_SPEED = 0.2
    KI_SPEED = 0.05
    INTEGRAL_MAX = 5.0
    
    # 横向 PID
    KP_TURN = 0.8                # 更灵敏
    KD_TURN = 0.1                # 少阻尼
    LOOK_DIST_BASE = 2.5         # 较长前瞻
    LOOK_SCALE = 0.15            # 随速增长快
```

---

## 六、常见现象与调参策略

### 现象 1: 车速忽高忽低

**原因**: 纵向 PID 响应太快或太慢

**解决方案**:
```python
# 过快震荡
KP_SPEED -= 0.05

# 过慢爬坡无力
KP_SPEED += 0.05
```

### 现象 2: 总是偏向参考线一侧

**原因**: 系统偏差未被消除

**解决方案**:
```python
# 如果总是偏左
KI_SPEED += 0.01    # 积分增强自动修正

# 或者检查：车辆是否未对准参考线起点
# 可能需要调整初始航向 heading
```

### 现象 3: 弯道偏离严重

**原因**: 横向响应不够或前瞻太远

**解决方案**:
```python
# 增加转向灵敏度
KP_TURN += 0.1

# 缩短前瞻距离（低速时）
if speed_ms < 5:
    LOOK_DIST_BASE = 1.5
```

### 现象 4: 直线行驶时方向盘抖动

**原因**: 横向控制过于灵敏

**解决方案**:
```python
# 增加微分项抑制抖动
KD_TURN += 0.05

# 或降低 P 增益
KP_TURN -= 0.05

# 加长前瞻距离
LOOK_DIST_BASE += 0.3
```

### 现象 5: 急弯失控冲出道路

**原因**: 前瞻距离太长或转向不足

**解决方案**:
```python
# 缩短前瞻（紧急）
LOOK_DIST_BASE = 1.0

# 同时大幅降低速度
TARGET_SPEED = 10.0

# 等稳定后再逐步提高
```

---

## 七、实际调试流程

### Phase 1: 静态测试（不动）

```python
# 检查代码能否运行
TARGET_SPEED = 0
执行一下
看 CSV 中所有值为 0
```

### Phase 2: 低速直线（20km/h）

```python
TARGET_SPEED = 20.0
KP_TURN = 0.5
KD_TURN = 0.1
LOOK_DIST_BASE = 1.5

观察 BEV 图：
- 如果直线走得好 → 继续 Phase 3
- 如果跑偏 → 调整 KP_TURN↑
- 如果抖动 → 调整 KD_TURN↑
```

### Phase 3: 加入弯道

```python
# 运行整个 loop 地图
观察弯道情况：
- 进入弯道时是否及时转向 → KP_TURN 够不够
- 出弯时是否甩尾 → KD_TURN 要不要加大
- 是否经常靠近边界 → LOOK_DIST 太长了
```

### Phase 4: 逐步提速

```python
speed_test = [20, 25, 30, 35]
for v in speed_test:
    TARGET_SPEED = v
    run_and_record()
    
    if lateral_error > 1.0m:
        print(f"{v}km/h 太高，需要调整参数")
        KP_TURN -= 0.05
        LOOK_DIST_BASE += 0.3
        
    if deviation 超过阈值:
        break
```

### Phase 5: 优化细节

根据数据反馈微调：

```python
# 从 CSV 分析：
avg_lat_err = mean(log['lateral_err_m'])
max_steering = max(abs(log['steering']))

if avg_lat_err > 0.5:
    KP_TURN += 0.02
elif max_steering > 0.5:
    KP_TURN -= 0.02
    KD_TURN += 0.02
```

---

## 八、参数调整速查表

| 问题 | 调整参数 | 调整方向 | 预期效果 |
|------|---------|---------|----------|
| 加速慢 | KP_SPEED | ↑ | 油门响应更快 |
| 速度震荡 | KP_SPEED | ↓ | 油门更平稳 |
| 始终偏低速 | KI_SPEED | ↑ | 自动补偿偏差 |
| 弯道偏离 | KP_TURN | ↑ | 转向更积极 |
| 弯道切内圈 | LOOK_DIST | ↓ | 更早发现弯道 |
| 直线抖动 | KD_TURN | ↑ | 抑制振荡 |
| 急转弯失控 | LOOK_DIST | ↓ | 缩短前瞻 |
| 高速不稳 | LOOK_DIST | ↑ | 提前预判 |

---

## 九、总结：参数设置的哲学

### 核心原则：

1. **安全第一** → 先从保守参数开始
2. **循序渐进** → 一次只调一个参数
3. **观察反馈** → 看 BEV 图和 CSV 数据
4. **理解物理** → 每个参数都有明确意义

### 最佳实践：

- **调试时用日志**：`SAVE_CSV=True`
- **可视化很重要**：BEV 视角能直观看到问题
- **记录每次调整**：方便回退和比较
- **分阶段验证**：低速→高速，直道→弯道

---

## 十、参考资源

### 理论阅读：

1. PID 控制基础：https://control.asu.edu/lectures/
2. Pure Pursuit 论文："Pure Pursuit: A Path Following Algorithm for Autonomous Vehicles"
3. 汽车动力学：https://en.wikipedia.org/wiki/Vehicle_dynamics

### 实践建议：

- 先在仿真环境中调试
- 再用真实车辆测试
- 逐步增加复杂度（加雨、加车、加坡道）

---

**最后提醒：**  
PID 参数没有"绝对正确"，只有"相对合适"。需要根据具体车辆特性、路面条件、行驶速度来调整。以上是经验和理论的总结，希望能帮助你快速上手！
