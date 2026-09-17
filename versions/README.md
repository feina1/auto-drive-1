# 代码版本说明

## V1 - PID Baseline (0913~0914)

PID + Pure Pursuit 路径跟踪控制器，无神经网络。

核心文件:
- `simple_track.py` — 闭环赛道 (Straight + Curve 拼接)
- `pid_demo_autonomous.py` — PID 控制器 + PathFollower
- `record_pid_images.py` — 数据录制 (图片 + 控制日志)
- `generate_replay.py` — HTML 回放页面

## V2 - DINOv3 First Attempt (0915)

单帧 RGB → DINOv3 (frozen) → MLP → 20 waypoints。40 个样本训练。

核心文件:
- `collect_data.py` — 数据收集: 随机偏移 → PID 恢复 → 记录 (图片, GT)
- `model_v2.py` — DINOv3 ViT-S/16 (frozen) + WaypointHead (MLP)
- `train_v2.py` — 训练脚本: 加权 L2 loss, AdamW, CosineAnnealing
- `run.py` — 闭环测试: 模型推理 → PID 跟踪预测路径

结果: 离线 val_l2=1.14m, 闭环存活 37 帧, Mean L2=5.3m

## V3 - DINOv3 2000 Samples (0916)

同 V2 架构，数据量扩大到 2000 个样本。

额外文件:
- `train_history.json` — 训练历史 (300 epochs)

结果: 离线 val_l2=0.24m, 闭环存活 1301 帧, Mean L2=1.48m

## V4 - Disturbance Injection (0916~0917)

在 V3 基础上改进数据收集:
- 扰动幅度加大: ±3m→±5m, ±15°→±30°
- GT 只取恢复段 (前 30 步)，避免 PID 收敛后变成正常行驶

核心文件:
- `collect_data.py` — 修改扰动参数 + 轨迹截断
- `test_variations.py` — 多驾驶风格 × 多赛道闭环测试脚本
