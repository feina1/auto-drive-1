# 参考路径生成说明

## 一、完整流程

```bash
# 1. 从 simple_track.py 提取参考路径
python3 extract_reference_loop.py

# 2. 生成两个文件：
    - reference_path_with_vehicles.png  (可视化图)
    - simple_track_reference_clean.json (数据文件，供控制算法使用)
```

---

## 二、核心代码位置

### **主入口**
- **文件**: `extract_reference_loop.py`
- **主要函数**: `main()` 和 `extract_single_clockwise_loop(env)`
- **功能**: 
  - 初始化 SimpleTrackEnv 环境
  - 遍历 road_network.graph 找到所有连接的车道段
  - 采样每条车道中心线点
  - 串联成完整闭环路径
  - 输出 JSON+PNG

### **依赖的环境类**
- **SimpleTrackMap**: 在 `simple_track.py` 中定义
  - `TRACK_SEGMENTS`: 赛道几何定义（直线 + 曲线）
  - `SimpleTrackEnv`: 创建仿真环境
  
- **road_network.graph**: MetaDrive 内部数据结构
  ```
  {
    NODE_A: {
      NODE_B: [lane1, lane2, ...]
    },
    ...
  }
  ```

### **输出数据处理**
- **Path**: `export_vehicle_markers()` 函数
  - 计算每 30m 的车辆标记位置
  - 记录 x, y, heading, s_distance

---

## 三、参数说明

### **关键配置（在 extract_reference_loop.py 中）**

```python
class Config:
    LANE_NUM = 2         # 每个方向 2 条车道 → 总共 4 车道
    LANE_WIDTH = 3.5     # 车道宽度（米）
    
    TRACK_SEGMENTS = [
        ("straight", 1000.0),   # 直线段
        ("curve", 500.0, 180, RIGHT),  # 弯道：半径 500m，转 180 度
        ...
    ]
```

### **输出文件格式**

**JSON 结构** (`simple_track_reference_clean.json`):
```json
{
  "metadata": {
    "total_path_length_m": 26185.8,
    "num_points": 26266,
    "vehicle_marker_interval_m": 30.0,
    "num_vehicles": 871
  },
  "reference_path": {
    "x": [10.0, 11.0, 12.0, ...],      // X 坐标数组
    "y": [0.0, 0.0, 0.0, ...],         // Y 坐标数组  
    "s_distances": [0.0, 1.0, 2.0, ...] // 累计距离
  },
  "vehicle_markers": [
    {
      "index": 10,
      "distance_m": 30.0,
      "x": 40.0,
      "y": 3.5,
      "heading_rad": 0.0,
      "heading_deg": 0.0
    },
    // ... 871 个标记点
  ]
}
```

---

## 四、调试要点

### **如果生成的路径有问题**

1. **路径不闭合**
   - 检查 `TRACK_SEGMENTS` 是否对称
   - 运行 `export_track_png()` 函数看布局

2. **路径长度不对**
   - 查看 `estimate_track_length()` 函数计算的理论长度
   - 比较与实际的差异

3. **有多余的分叉**
   - 修改 `extract_single_clockwise_loop()` 中的节点遍历逻辑
   - 确保只走单向路径

### **验证路径是否正确**

```python
import json

with open("simple_track_reference_clean.json") as f:
    data = json.load(f)

print(f"路径点数：{len(data['reference_path']['x'])}")
print(f"总长度：{data['metadata']['total_path_length_m']:.1f} m")
print(f"车辆标记：{data['metadata']['num_vehicles']} 个")

# 检查首尾是否接近起点
start_x = data['reference_path']['x'][0]
end_x = data['reference_path']['x'][-1]
print(f"起点 X: {start_x:.2f}, 终点 X: {end_x:.2f}, 差值：{abs(end_x-start_x):.2f}")
```

---

## 五、常用命令

### **生成参考路径（最简）**
```bash
cd /Users/wl/Downloads/wlwork/auto-drive-1
python3 extract_reference_loop.py
```

### **只看路径统计（不调用环境）**
```bash
python3 -c "
import json
with open('simple_track_reference_clean.json') as f:
    d = json.load(f)
print(d['metadata'])
"
```

### **重新生成 PNG 可视化**
```bash
# 直接重跑 extract_reference_loop.py 即可
# 会自动保存为 reference_path_with_vehicles.png
```

---

## 六、相关文件清单

| 文件 | 作用 | 状态 |
|------|------|------|
| `extract_reference_loop.py` | **核心脚本，生成参考路径** | ✅ 可用 |
| `simple_track.py` | 赛道定义和环境类 | ✅ 已存在 |
| `simple_track_reference_clean.json` | 生成的参考路径数据 | ✅ 已生成 |
| `reference_path_with_vehicles.png` | 可视化图像 | ✅ 已生成 |
| `pid_control_test.py` | PID 控制器测试代码 | ⚠️ 需集成到真实环境 |
| `drive_simple_track.py` | 仿真驱动脚本 | ⚠️ 需接入控制器 |

---

## 七、常见问题 FAQ

**Q: 为什么每次都要重新生成？**  
A: 因为 `SimpleTrackMap` 是动态生成的，除非固定 seed，否则每次不同。建议在 `extract_reference_loop.py` 中设置固定 seed=0。

**Q: JSON 文件大小有 1.9MB 是否正常？**  
A: 正常。包含了 26,266 个点的完整路径，压缩后约 100KB。

**Q: 可以调整车辆标记间隔吗？**  
A: 在 `export_vehicle_markers()` 函数中修改 `interval` 参数即可，默认为 30m。

**Q: 如果我要用其他赛道怎么办？**  
A: 需要：
1. 修改 `simple_track.py` 的 `TRACK_SEGMENTS`
2. 重新运行 `extract_reference_loop.py`

---

## 八、推荐工作流

```
1. 修改赛道设计 → simple_track.py
2. 运行提取脚本 → extract_reference_loop.py
3. 检查 JSON 数据 → head -20 simple_track_reference_clean.json
4. 打开 PNG 可视化 → reference_path_with_vehicles.png
5. 确认无误后 → 写入控制代码 → drive_simple_track.py
```
