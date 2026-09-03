# PID 自动驾驶控制器 — 代码说明与原理

## 1. 整体架构

```
┌─────────────────────────────────────────────────┐
│                  主循环 (while True)              │
│                                                   │
│  读取车辆状态 → PathFollower.compute_control()   │
│       ↓                    ↓                      │
│  [steering, throttle]  →  env.step(action)        │
│       ↓                                           │
│  env.render() 显示 3D 画面                        │
│       ↓                                           │
│  LapCounter 计圈 → 到终点重置导航                  │
└─────────────────────────────────────────────────
```

文件结构：
- `PIDController` — 通用 PID 控制器
- `PathFollower` — 路径跟踪控制器（Pure Pursuit + PID）
- `BEVVisualizer` — matplotlib 鸟瞰图（调试用，3D 模式下不使用）
- `export_map_jpg` — 导出地图 JPG（调试用，3D 模式下不使用）
- 主函数 — 创建环境、控制器，运行循环

---

## 2. PID 控制器原理

### 2.1 什么是 PID？

PID 是工业控制中最经典的反馈控制算法，由三部分组成：

```
output = Kp × error + Ki × ∫error·dt + Kd × d(error)/dt
          ────────    ──────────────    ──────────────
           比例项 P      积分项 I          微分项 D
```

| 项 | 作用 | 直观理解 |
|---|---|---|
| **P (比例)** | 误差越大，输出越大 | "差多少补多少" |
| **I (积分)** | 累积历史误差，消除稳态偏差 | "一直偏就慢慢加大修正" |
| **D (微分)** | 误差变化趋势，抑制振荡 | "变化太快就踩刹车" |

### 2.2 代码实现

```python
class PIDController:
    def __init__(self, kp, ki, kd, integral_max=5.0):
        self.kp = kp          # 比例增益
        self.ki = ki          # 积分增益
        self.kd = kd          # 微分增益
        self.integral_max = integral_max  # 积分限幅（防积分饱和）
        self.integral = 0.0   # 积分累积值
        self.last_error = None  # 上一次的误差（用于算微分）

    def update(self, error, dt):
        p_term = self.kp * error                        # 比例项
        self.integral += error * dt                     # 累积误差
        self.integral = clip(integral, -max, +max)      # 限幅
        i_term = self.ki * self.integral                # 积分项
        d_term = self.kd * (error - last_error) / dt    # 微分项
        self.last_error = error
        return p_term + i_term + d_term
```

**积分限幅 (anti-windup)**：如果不限制积分值，当误差长期存在时积分会无限增长，导致控制器"反应过度"。

---

## 3. Pure Pursuit 路径跟踪原理

### 3.1 核心思想

Pure Pursuit（纯追踪）模仿人开车的直觉：**不要看车头正前方，要看前方一段距离的目标点，朝那个点开。**

```
        参考路径:  ●──●──●──●──●──●──●──●──●
                              ↑
                           前瞻点 (lookahead)
                          /
                         /  朝这个点开
                        /
        车辆位置:  🚗
```

### 3.2 前瞻距离

前瞻距离不是固定的，而是随速度变化：

```
lookahead_dist = look_dist_base + look_dist_scale × speed
               = 8.0 + 0.3 × speed_ms
```

- 低速时：前瞻距离短 → 反应灵敏，贴紧路径
- 高速时：前瞻距离长 → 提前转弯，避免甩尾

### 3.3 找前瞻点的过程

```
1. 在参考路径上找到离车最近的点 (closest_idx)
2. 从 closest_idx 往前遍历路径点
3. 找到第一个距离车辆 ≥ lookahead_dist 的点 → 这就是前瞻点
```

```python
def find_lookahead_point(self, x, y, speed_ms):
    lookahead_dist = 8.0 + 0.3 * speed_ms
    closest_idx = self.find_closest_idx(x, y)  # 最近点

    for i in range(closest_idx + 1, closest_idx + 1000):
        dist = distance(path[i], vehicle)
        if dist >= lookahead_dist:
            return path[i]  # 找到前瞻点
```

---

## 4. 关键设计：为什么 PID 输入要加横向误差？

### 4.1 问题

如果 PID 的输入**只有航向误差**，会出现这种情况：

```
参考路径:  ────────────────────
                          ↑
                        车在这里
                          ↑
                    车头已经对准路径方向
                    航向误差 ≈ 0
                    PID 输出 ≈ 0
                    → 车不再修正，一直偏着走！
```

### 4.2 解决方案

把**横向偏移**也加入 PID 输入，转换成等效的角度修正：

```python
# 横向偏移修正: 用 arctan 把距离转换成角度
lateral_correction = -arctan(signed_lat_err / lookahead_dist)

# 综合误差 = 航向误差 + 横向修正
combined_error = heading_err + lateral_correction

# PID 根据综合误差输出转向
steering = PID.update(combined_error, dt)
```

**效果**：即使车头对准了路径方向，只要车还偏在一边，`lateral_correction` 就不为零，PID 会继续修正直到车回到路径上。

### 4.3 有符号横向误差

横向误差不是简单的距离，而是**有方向的**：

```python
# 路径方向的单位法向量（指向左侧）
nx = -path_dy / path_len
ny =  path_dx / path_len

# 车辆到路径点的向量 投影到法向量
signed_lat = (vehicle - path_point) · (nx, ny)

# signed_lat > 0 → 车在路径左侧 → 需要右转
# signed_lat < 0 → 车在路径右侧 → 需要左转
```

---

## 5. 纵向速度控制

纵向 PID 很简单：目标是保持 70 km/h。

```python
speed_error = target_speed - current_speed   # 正=太慢，负=太快
throttle = PID.update(speed_error, dt)        # 输出 -1~1
# throttle > 0 → 加油
# throttle < 0 → 刹车
```

---

## 6. 主循环流程

```python
while True:
    # 1. 读取车辆状态
    x, y = env.agent.position[:2]
    heading = env.agent.heading_theta    # 航向角（弧度）
    speed = env.agent.speed_km_h          # 速度（km/h）

    # 2. 计算控制指令
    [steering, throttle] = follower.compute_control(x, y, heading, speed, dt)

    # 3. 执行控制
    env.step([steering, throttle])

    # 4. 计圈
    laps = lap_counter.update(position)
    if laps 增加了:
        env.agent.reset_navigation()  # 重置导航，继续下一圈

    # 5. 渲染画面
    env.render(text={速度, 转向, 油门, 横向误差, 圈数})
```

---

## 7. 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `kp` (转向) | 0.8 | 比例增益，越大转向越猛 |
| `ki` (转向) | 0.08 | 积分增益，消除稳态偏差 |
| `kd` (转向) | 0.15 | 微分增益，抑制振荡 |
| `look_dist_base` | 8.0 m | 基础前瞻距离 |
| `look_dist_scale` | 0.3 | 速度相关的前瞻系数 |
| `max_steer` | 0.6 | 最大转向角（约 34°） |
| `TARGET_SPEED_KMH` | 70.0 | 目标速度 |
| `dt` | 0.1 s | 控制周期 |

---

## 8. 实验结果

| 指标 | 数值 |
|------|------|
| 稳定圈数 | 5+ 圈（可无限跑） |
| 平均横向误差 | **0.22 m** |
| 最大横向误差 | **1.19 m**（弯道处） |
| 5m 偏移恢复 | **200 帧内恢复到 <0.2m** |
| 速度稳定性 | 70 ± 0.5 km/h |

---

## 9. 相关文件

| 文件 | 作用 |
|------|------|
| `pid_demo_autonomous.py` | 本文件，PID 自动驾驶主程序 |
| `simple_track.py` | 自定义赛道环境（闭合环形） |
| `simple_track_reference_clean.json` | 参考路径（内侧车道中心线，26266 个点） |
| `runtime_config.py` | 渲染配置、后端配置 |
| `drive_simple_track.py` | 参考代码（人工/PPO 驾驶） |
| `pid_control_log.csv` | 运行日志（自动保存） |

---

## 10. 运行方式

```bash
# 3D 可视化模式（默认）
python3 pid_demo_autonomous.py

# 按 ESC 退出
```
