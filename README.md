# auto-drive-1

基于 MetaDrive 的自动驾驶仿真：自定义闭环赛道 + PID 控制 + 图像录制回放。

## 安装

```bash
./install.sh
```

创建 conda 环境 `autodrive` (Python 3.10)，安装 MetaDrive（本地源码）和所有依赖，下载 3D 资源。

## 使用

```bash
conda activate autodrive

python pid_demo_autonomous.py       # PID 自动驾驶
python record_pid_images.py         # PID 录制图像
python record.py                    # Expert 录制
python drive_simple_track.py        # 简单赛道驾驶
python generate_replay.py           # 生成回放 HTML
```

> 每次开新终端都要先 `conda activate autodrive`。
