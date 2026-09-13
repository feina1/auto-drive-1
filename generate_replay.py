#!/usr/bin/env python3
"""生成 PID 录制回放 HTML 页面

用法:
    python3 generate_replay.py                              # 自动找最新录制
    python3 generate_replay.py records_pid/2026-09-06_17-11  # 指定会话
    python3 generate_replay.py --no-open                     # 不自动打开浏览器

生成 replay.html 在同一会话目录下，双击即可在浏览器中打开回放。
"""

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REF_PATH_FILE = PROJECT_DIR / "simple_track_reference_clean.json"


def find_latest_session(records_root: Path) -> Path:
    """找到 records_root 下最新的会话目录。"""
    if not records_root.exists():
        raise FileNotFoundError(f"录制目录不存在: {records_root}")
    sessions = sorted(records_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sessions:
        raise FileNotFoundError(f"录制目录为空: {records_root}")
    return sessions[0]


def load_session(session_dir: Path):
    """加载会话的 metadata、frames、参考路径。"""
    metadata_path = session_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json 不存在: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    frames_dir = session_dir / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"frames 目录不存在: {frames_dir}")

    frame_files = sorted(frames_dir.glob("*.json"))
    if not frame_files:
        raise FileNotFoundError(f"frames 目录为空: {frames_dir}")

    frames = []
    for ff in frame_files:
        with open(ff) as f:
            data = json.load(f)
        frames.append({
            "x": data["position"]["x"],
            "y": data["position"]["y"],
            "heading": data["heading"],
            "image": data["image"],
        })

    # 加载参考路径并降采样
    if not REF_PATH_FILE.exists():
        raise FileNotFoundError(f"参考路径文件不存在: {REF_PATH_FILE}")
    with open(REF_PATH_FILE) as f:
        ref_data = json.load(f)
    ref_x = ref_data["reference_path"]["x"]
    ref_y = ref_data["reference_path"]["y"]
    step = max(1, len(ref_x) // 3000)
    ref_x_ds = ref_x[::step]
    ref_y_ds = ref_y[::step]

    return metadata, frames, ref_x_ds, ref_y_ds


def detect_lap_frames(frames, finish_radius=40.0, min_travel_m=5000.0):
    """检测每次回到起点附近的帧索引 (完成一圈的点)."""
    if not frames:
        return []
    origin_x, origin_y = frames[0]["x"], frames[0]["y"]
    lap_frames = []  # 每圈结束时的帧索引
    travel_m = 0.0
    inside_finish = True
    last_x, last_y = origin_x, origin_y
    for i, f in enumerate(frames):
        x, y = f["x"], f["y"]
        travel_m += ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
        last_x, last_y = x, y
        dist = ((x - origin_x) ** 2 + (y - origin_y) ** 2) ** 0.5
        if dist >= finish_radius:
            inside_finish = False
        elif not inside_finish and travel_m >= min_travel_m:
            lap_frames.append(i)
            travel_m = 0.0
            inside_finish = True
    return lap_frames


def generate_html(session_dir: Path, metadata, frames, ref_x, ref_y, lap_frames) -> str:
    """生成自包含的 HTML 回放页面。"""
    # 计算世界坐标范围 (参考路径 + 轨迹)
    all_x = ref_x + [f["x"] for f in frames]
    all_y = ref_y + [f["y"] for f in frames]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    margin = 50.0
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin

    total_laps = len(lap_frames) + 1  # 完成的圈数 + 当前正在跑的圈
    # 每圈结束时间 (秒)
    lap_times = [round(f / metadata.get("record_hz", 10), 1) for f in lap_frames]

    # 数据 JSON (紧凑格式)
    data_json = json.dumps({
        "fps": metadata.get("record_hz", 10),
        "total_laps": total_laps,
        "lap_frames": lap_frames,
        "lap_times": lap_times,
        "ref": {"x": ref_x, "y": ref_y},
        "frames": frames,
        "bounds": [x_min, x_max, y_min, y_max],
    }, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>PID 录制回放 - {session_dir.name}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a1a; color: #eee; }}
.header {{ background: #2a2a2a; padding: 12px 20px; border-bottom: 1px solid #444; display: flex; align-items: center; gap: 16px; }}
.header h1 {{ font-size: 16px; font-weight: 500; }}
.header .info {{ font-size: 13px; color: #aaa; }}
.main {{ display: flex; height: calc(100vh - 57px); }}
.left {{ flex: 1; display: flex; flex-direction: column; background: #000; }}
.right {{ width: 45%; display: flex; flex-direction: column; background: #222; border-left: 1px solid #444; }}
.img-container {{ flex: 1; display: flex; align-items: center; justify-content: center; padding: 20px; }}
.img-container img {{ max-width: 100%; max-height: 100%; object-fit: contain; background: #111; }}
.controls {{ background: #2a2a2a; padding: 12px 20px; border-top: 1px solid #444; }}
.controls-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
.controls-row:last-child {{ margin-bottom: 0; }}
button {{ background: #4a4a4a; color: #fff; border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 13px; }}
button:hover {{ background: #5a5a5a; }}
button.active {{ background: #4a9eff; }}
input[type="range"] {{ flex: 1; }}
select {{ background: #4a4a4a; color: #fff; border: none; padding: 4px 8px; border-radius: 4px; font-size: 13px; }}
.label {{ font-size: 12px; color: #aaa; min-width: 60px; }}
.value {{ font-size: 13px; color: #fff; font-family: monospace; }}
.canvas-container {{ flex: 1; display: flex; align-items: center; justify-content: center; padding: 20px; }}
canvas {{ background: #111; border-radius: 4px; }}
.legend {{ padding: 12px 20px; font-size: 12px; color: #aaa; border-top: 1px solid #444; }}
.legend span {{ display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: middle; }}
</style>
</head>
<body>
<div class="header">
  <h1>PID 录制回放</h1>
  <div class="info">{session_dir.name} | {len(frames)} 帧 | {metadata.get('record_hz', 10)} FPS | {total_laps} 圈</div>
</div>
<div class="main">
  <div class="left">
    <div class="img-container">
      <img id="frame-img" src="" alt="Frame">
    </div>
    <div class="controls">
      <div class="controls-row">
        <button id="btn-play">▶ 播放</button>
        <button id="btn-prev">⏮</button>
        <button id="btn-next"></button>
        <span class="label">速度:</span>
        <select id="speed-select">
          <option value="0.25">0.25x</option>
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
          <option value="10">10x</option>
          <option value="20">20x</option>
          <option value="30">30x</option>
        </select>
      </div>
      <div class="controls-row">
        <span class="label">进度:</span>
        <input type="range" id="progress-slider" min="0" max="{len(frames)-1}" value="0">
      </div>
      <div class="controls-row">
        <span class="label">帧:</span>
        <span class="value" id="frame-label">0 / {len(frames)-1}</span>
        <span class="label" style="margin-left:20px;">位置:</span>
        <span class="value" id="pos-label">-</span>
        <span class="label" style="margin-left:20px;">圈:</span>
        <span class="value" id="lap-label">1/{total_laps}</span>
      </div>
    </div>
  </div>
  <div class="right">
    <div class="canvas-container">
      <canvas id="bev-canvas" width="600" height="600"></canvas>
    </div>
    <div class="legend">
      <span style="background:#4a9eff;"></span>参考路径
      <span style="background:#ffd700; margin-left:12px; opacity:0.6;"></span>完整轨迹
      <span style="background:#ff4a4a; margin-left:12px;"></span>已播放
      <span style="background:#ff4a4a; margin-left:12px; border-radius:50%;"></span>当前位置
      <span style="background:#00ff88; margin-left:12px; font-size:14px;">&#9733;</span>圈结束点
    </div>
  </div>
</div>
<script>
const DATA = {data_json};
const frames = DATA.frames;
const refPath = DATA.ref;
const bounds = DATA.bounds;
const fps = DATA.fps;
const totalLaps = DATA.total_laps;
const lapFrames = DATA.lap_frames;
const lapTimes = DATA.lap_times;

let currentFrame = 0;
let playing = false;
let speed = 1;
let lastTime = 0;
let accumulator = 0;

const imgEl = document.getElementById('frame-img');
const slider = document.getElementById('progress-slider');
const frameLabel = document.getElementById('frame-label');
const posLabel = document.getElementById('pos-label');
const lapLabel = document.getElementById('lap-label');
const btnPlay = document.getElementById('btn-play');
const canvas = document.getElementById('bev-canvas');
const ctx = canvas.getContext('2d');

// 世界坐标 → 画布坐标
function worldToCanvas(wx, wy) {{
  const [xMin, xMax, yMin, yMax] = bounds;
  const w = canvas.width;
  const h = canvas.height;
  const scale = Math.min(w / (xMax - xMin), h / (yMax - yMin));
  const cx = (xMin + xMax) / 2;
  const cy = (yMin + yMax) / 2;
  const px = (wx - cx) * scale + w / 2;
  const py = h / 2 - (wy - cy) * scale;
  return [px, py];
}}

// 绘制 BEV
function drawBEV() {{
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // 绘制参考路径
  ctx.strokeStyle = '#4a9eff';
  ctx.lineWidth = 2;
  ctx.globalAlpha = 0.6;
  ctx.beginPath();
  for (let i = 0; i < refPath.x.length; i++) {{
    const [px, py] = worldToCanvas(refPath.x[i], refPath.y[i]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }}
  ctx.stroke();
  ctx.globalAlpha = 1.0;

  // 绘制完整录制轨迹 (黄色，所有帧)
  if (frames.length > 1) {{
    ctx.strokeStyle = '#ffd700';
    ctx.lineWidth = 3;
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    for (let i = 0; i < frames.length; i++) {{
      const [px, py] = worldToCanvas(frames[i].x, frames[i].y);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }}
    ctx.stroke();
    ctx.globalAlpha = 1.0;
  }}
  
  // 绘制已播放轨迹 (红色，到当前帧)
  if (currentFrame > 0) {{
    ctx.strokeStyle = '#ff4a4a';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i <= currentFrame; i++) {{
      const [px, py] = worldToCanvas(frames[i].x, frames[i].y);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }}
    ctx.stroke();
  }}

  // 绘制圈结束点 (绿色星号)
  for (let li = 0; li < lapFrames.length; li++) {{
    const fi = lapFrames[li];
    const [lx, ly] = worldToCanvas(frames[fi].x, frames[fi].y);
    ctx.fillStyle = '#00ff88';
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('\u2605', lx, ly);
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 10px sans-serif';
    ctx.fillText('' + (li + 1), lx, ly + 12);
  }}

  // 绘制当前位置红点
  const [cx, cy] = worldToCanvas(frames[currentFrame].x, frames[currentFrame].y);
  ctx.fillStyle = '#ff4a4a';
  ctx.beginPath();
  ctx.arc(cx, cy, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.stroke();

  // 绘制朝向指示线
  const heading = frames[currentFrame].heading;
  const lineLen = 20;
  const ex = cx + Math.cos(heading) * lineLen;
  const ey = cy - Math.sin(heading) * lineLen;
  ctx.strokeStyle = '#ff4a4a';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(ex, ey);
  ctx.stroke();
}}

// 更新显示
function updateDisplay() {{
  imgEl.src = frames[currentFrame].image;
  slider.value = currentFrame;
  frameLabel.textContent = `${{currentFrame}} / ${{frames.length - 1}}`;
  posLabel.textContent = `(${{frames[currentFrame].x.toFixed(1)}}, ${{frames[currentFrame].y.toFixed(1)}})`;
  // 计算当前圈数 (第几圈, 从1开始)
  let curLap = 1;
  for (let i = 0; i < lapFrames.length; i++) {{
    if (currentFrame >= lapFrames[i]) curLap = i + 2;
  }}
  lapLabel.textContent = curLap + '/' + totalLaps;
  drawBEV();
}}

// 播放循环
function playLoop(timestamp) {{
  if (!playing) return;
  if (!lastTime) lastTime = timestamp;
  const delta = timestamp - lastTime;
  lastTime = timestamp;
  accumulator += delta * speed;
  const frameInterval = 1000 / fps;
  while (accumulator >= frameInterval) {{
    accumulator -= frameInterval;
    currentFrame++;
    if (currentFrame >= frames.length) {{
      currentFrame = frames.length - 1;
      playing = false;
      btnPlay.textContent = '▶ 播放';
      break;
    }}
    updateDisplay();
  }}
  requestAnimationFrame(playLoop);
}}

// 事件绑定
btnPlay.addEventListener('click', () => {{
  playing = !playing;
  btnPlay.textContent = playing ? ' 暂停' : '▶ 播放';
  if (playing) {{
    lastTime = 0;
    accumulator = 0;
    requestAnimationFrame(playLoop);
  }}
}});

document.getElementById('btn-prev').addEventListener('click', () => {{
  if (currentFrame > 0) {{
    currentFrame--;
    updateDisplay();
  }}
}});

document.getElementById('btn-next').addEventListener('click', () => {{
  if (currentFrame < frames.length - 1) {{
    currentFrame++;
    updateDisplay();
  }}
}});

slider.addEventListener('input', (e) => {{
  currentFrame = parseInt(e.target.value);
  updateDisplay();
}});

document.getElementById('speed-select').addEventListener('change', (e) => {{
  speed = parseFloat(e.target.value);
}});

// 键盘控制
document.addEventListener('keydown', (e) => {{
  if (e.code === 'Space') {{
    e.preventDefault();
    btnPlay.click();
  }} else if (e.code === 'ArrowLeft') {{
    document.getElementById('btn-prev').click();
  }} else if (e.code === 'ArrowRight') {{
    document.getElementById('btn-next').click();
  }}
}});

// 初始化
updateDisplay();
</script>
</body>
</html>"""
    return html


def main():
    auto_open = "--no-open" not in sys.argv
    session_arg = None
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        session_arg = arg

    if session_arg:
        session_dir = Path(session_arg).resolve()
    else:
        records_root = PROJECT_DIR / "records_pid"
        session_dir = find_latest_session(records_root)

    print(f"会话目录: {session_dir}")
    metadata, frames, ref_x, ref_y = load_session(session_dir)
    print(f"加载 {len(frames)} 帧 | 参考路径 {len(ref_x)} 点")

    lap_frames = detect_lap_frames(frames)
    if lap_frames:
        hz = metadata.get("record_hz", 10)
        print(f"检测到 {len(lap_frames)} 圈:")
        for i, fi in enumerate(lap_frames):
            print(f"  第 {i+1} 圈结束: frame {fi} (t={fi/hz:.0f}s)")
    else:
        print("未检测到完整圈数")

    html = generate_html(session_dir, metadata, frames, ref_x, ref_y, lap_frames)
    output_path = session_dir / "replay.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✓ 生成回放页面: {output_path}")

    if auto_open:
        import subprocess, platform
        cmd = "open" if platform.system() == "Darwin" else "xdg-open"
        subprocess.run([cmd, str(output_path)], check=False)
        print(f"已在浏览器中打开")


if __name__ == "__main__":
    main()
