"""Controller 验证: 参考线 → 滑动窗口 waypoints → 控制器 → (steer, throttle).

要求见同目录 Controller验证方案.md, 四条已锁定决策:
  D1 waypoints = 10 点, 不含 t 本身 (折线原点由控制器内部补)
  D2 采样时刻 τ_k = t + 0.5k 连续滑动, 禁止对齐到 0.5 网格
  D3 来源 = 参考线几何 + 当前实测速度沿弧长推进 (不回放专家时间轴)
  D4 控制频率 10 Hz (dt = 0.1 s)

参考线完全不变, 不做任何横向偏移/车辆传送。既有模块一行不改, 只 import 复用。

用法:
    python3 validate_controller.py                 # 无头整圈
    python3 validate_controller.py --quick         # 前 90s, 先看时序断言
    python3 validate_controller.py --render        # 开窗口目视 (屏显为英文)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

import config
import traj_utils as tu
from controllers import GeometricController, path_from_waypoints, point_at_arc
from make_waypoints import world_to_ego

WP_DT = config.WP_DT      # 0.5 s  (2 Hz)
WP_N = config.WP_COUNT    # 10 点   (D1)
DT = config.SIM_DT        # 0.1 s   (D4: 10 Hz)

# ---- 判据阈值 (方案 §5) ----
T1_REL_MAX = 0.05         # ‖wp₀‖ vs v·0.5s 逐 tick 最大相对误差
T2_REL_MED_MAX = 0.05     # 相邻点间距 vs v·0.5s 中位相对误差
T3_REL_MAX = 0.10         # s_car 每 tick 增量 vs v·0.1s
T3_MONO_MIN_M = -0.05     # 里程不回退容差
T4A_MAX_M = 1e-6          # ego→world 反映射与生成点之差 (可逆性)
T4B_MAX_M = 0.30          # 车在 τ_k 实际位置对当时 wp_world[k] 的横向偏差
T5_Y_RES_MAX = 0.15       # 首点横向位置 vs 局部曲率预测 ½κd² 的残差
LAT_MEAN_MAX, LAT_P95_MAX, LAT_MAX_MAX = 0.05, 0.10, 0.25
SPEED_MEAN_MIN, SPEED_MIN_MIN = 68.0, 60.0
STEER_SAT_RATIO_MAX = 0.01
MIN_EXPECT_SPACING_M = 1.0    # 低速帧点距校验无意义, 跳过
WARMUP_TICKS = 5              # 前几个 tick 是 reset 后物理瞬态, 不进入时序断言
OFFROAD_TOL = 0.005           # 出界判据 = max(配置上限, 专家同窗基线 + 此容差)

# --- 叠加层样式 (仅 --render, 不参与任何判据) ---
SCOPE_WINDOW_S = 10.0         # steer/throttle 曲线回看窗口
SCOPE_STEER_FULL = 0.60       # steer 满量程显示 (rad, 与限幅一致)
SCOPE_THR_FULL = 1.00         # throttle 满量程
WORLD_POINT_Z = 0.7         # waypoint 球离地高度
# sphere.egg 是半径 1 的单位球, 而 make_point_drawer 内部只做 scale/10,
# 所以世界直径 = scale/5 米 —— 这里直接按“想要的直径米数”写, 不再拍脑袋
WP_POINT_DIAM_M = 0.55      # waypoint 球直径 (车宽 1.9 m, 点间距 ~9.7 m)
LOOK_POINT_DIAM_M = 0.95    # 前视点球直径 (比 waypoint 大一号以区分)


class ReferencePath:
    """参考线: 弧长均匀重采样 + 连续投影 + 闭环取点 (D3 的几何底座)."""

    def __init__(self, x, y, spacing: float = 1.0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        seg = np.hypot(np.diff(x), np.diff(y))
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        self.L = float(arc[-1])
        self.n = max(int(round(self.L / spacing)), 8)
        self.spacing = self.L / self.n
        s_grid = np.linspace(0.0, self.L, self.n, endpoint=False)
        self.px = np.interp(s_grid, arc, x)
        self.py = np.interp(s_grid, arc, y)
        nxt = (np.arange(self.n) + 1) % self.n          # 闭环: 末段回指首点
        dx = self.px[nxt] - self.px
        dy = self.py[nxt] - self.py
        ln = np.hypot(dx, dy) + 1e-12
        self.ux = dx / ln
        self.uy = dy / ln
        self.seg_len = ln
        self.cursor = 0

    def reset(self):
        self.cursor = 0

    def _search(self, x, y, idx):
        """在给定段索引上求最近点 (段内插值 → 连续里程, 不做量化)."""
        px, py = self.px[idx], self.py[idx]
        t = np.clip((x - px) * self.ux[idx] + (y - py) * self.uy[idx],
                    0.0, self.seg_len[idx])
        d2 = (px + self.ux[idx] * t - x) ** 2 + (py + self.uy[idx] * t - y) ** 2
        j = int(np.argmin(d2))
        return int(idx[j]), float(t[j]), float(d2[j])

    def project(self, xy):
        """车辆位置 → (连续里程 s, 段索引). 窗口 ±60m, 失配则全局兜底."""
        win = max(int(60.0 / self.spacing), 8)
        idx = np.mod(np.arange(self.cursor - win, self.cursor + win), self.n)
        best, t, d2 = self._search(xy[0], xy[1], idx)
        if d2 > 50.0 ** 2:
            best, t, _ = self._search(xy[0], xy[1], np.arange(self.n))
        self.cursor = best
        return best * self.spacing + t, best

    def point_at(self, s):
        """按里程取世界坐标, 数组输入, 闭环回绕."""
        sm = np.mod(np.asarray(s, dtype=float), self.L)
        idx = np.minimum((sm / self.spacing).astype(int), self.n - 1)
        t = sm - idx * self.spacing
        return np.stack([self.px[idx] + self.ux[idx] * t,
                         self.py[idx] + self.uy[idx] * t], axis=-1)

    def tangent_at(self, s):
        sm = np.mod(np.atleast_1d(np.asarray(s, dtype=float)), self.L)
        idx = np.minimum((sm / self.spacing).astype(int), self.n - 1)
        return np.arctan2(self.uy[idx], self.ux[idx])


def ego_to_world(pts_ego, ego_xy, heading):
    """world_to_ego 的解析逆 (独立写出, 用于 T4a 可逆性校验)."""
    c, s = np.cos(heading), np.sin(heading)
    out = np.empty_like(pts_ego)
    out[:, 0] = ego_xy[0] + pts_ego[:, 0] * c - pts_ego[:, 1] * s
    out[:, 1] = ego_xy[1] + pts_ego[:, 0] * s + pts_ego[:, 1] * c
    return out


def make_wp(ref: ReferencePath, ego_xy, heading: float, v_ms: float):
    """D2/D3 的实现: s_k = s_car + v·0.5k, k=1..10, 再转当前 ego 系.

    注意: 全程无取整/对齐 —— 滑动相位由 s_car 连续投影与实测速度天然保证。
    """
    s_car, _ = ref.project(ego_xy)
    s_k = s_car + v_ms * WP_DT * np.arange(1, WP_N + 1)
    wp_world = ref.point_at(s_k)
    return world_to_ego(wp_world, ego_xy, heading), wp_world, s_car, s_k


def build_render_env(fps=None):
    """有头演示环境: 与 traj_utils.build_headless_env 同配置, 仅换渲染开关.

    故意不改动既有模块, 故在此复制一份配置 (差异只有 use_render / show_interface)。
    fps=None 沿用 MetaDrive 默认节流 (约 1/physics 步长 ≈ 50 fps);
    传大值相当于解除节流, 用于快进到弯道取证。
    """
    from simple_track import LANE_NUM, LANE_WIDTH, SimpleTrackEnv
    from runtime_config import get_render_config

    cfg = dict(
        use_render=True,
        manual_control=False,
        traffic_density=0.0,
        num_scenarios=1,
        start_seed=0,
        map_region_size=4096,
        show_interface=False,
        show_logo=False,
        show_fps=True,
        force_render_fps=fps,
        **get_render_config(),
        random_lane_width=False,
        random_lane_num=False,
        out_of_route_done=False,
        on_continuous_line_done=False,
        out_of_road_done=False,
        crash_vehicle_done=False,
        crash_object_done=False,
        crash_human_done=False,
        map_config=dict(lane_num=LANE_NUM, lane_width=LANE_WIDTH),
        vehicle_config=dict(
            show_navi_mark=False,
            show_line_to_navi_mark=False,
            show_lidar=False,
            spawn_velocity=[config.TARGET_SPEED_KMH / 3.6, 0.0],
            spawn_velocity_car_frame=True,
        ),
    )
    return SimpleTrackEnv(cfg)


def snap_window(env, out_dir, t):
    """把当前窗口存图到 out_dir。

    ShowBase.screenshot 只能写当前工作目录且文件名带时间戳, 所以取最新一个改名。
    仅用于取证, 不影响任何判据与落盘数据。
    """
    env.engine.taskMgr.step()
    prefix = f"snap{int(round(t * 100)):05d}"
    env.engine.screenshot(namePrefix=prefix, defaultFilename=1)
    cands = sorted(Path(".").glob(f"{prefix}*.jpg"), key=lambda p: p.stat().st_mtime)
    if not cands:
        return None
    dst = Path(out_dir) / f"snap_t{t:07.1f}.jpg"
    cands[-1].replace(dst)
    return dst


def expert_offroad_baseline(traj_path, until_t: float) -> float:
    """专家录制轨迹在同一时间窗内的出界占比 (地图判据误报的基线)."""
    frames, _ = tu.load_trajectory(traj_path)
    flags = [bool(f.get("offroad", False)) for f in frames if f["t"] <= until_t]
    return float(np.mean(flags)) if flags else 0.0


class SceneOverlay:
    """有头模式下的实时叠加层 (只用于观察, 不参与控制, 也不写入任何判据数据)。

    3D 场景: 当前喂给控制器的 10 个 waypoints (绿折线 + 白刻度), 以及 Pure Pursuit
            实际前视点 (红绳 + 黄叉), 用来目视“控制器在看哪里”。
    屏内右下角: 过去 scope_window 秒的 steer / throttle 两条曲线 (滚动)。

    节点管理: 静态框/标签只建一次; 曲线与 waypoints 每 tick 重建并销毁上一帧的
    (整条折线共用一个 LineSegs 节点, 不是每段一个, 避免 10 Hz 下节点爆炸)。
    panda 相关 import 全部放在方法内, 保证 --self-test 不依赖渲染栈。
    """

    def __init__(self, env, scope_window=SCOPE_WINDOW_S):
        from panda3d.core import LineSegs, TextNode
        self._LineSegs = LineSegs
        self._TextNode = TextNode
        eng = env.engine
        self.aspect = eng.aspect2d
        self.ws = eng.w_scale
        self.hs = eng.h_scale
        self.win_s = float(scope_window)
        self._dyn = []
        self._static = []
        # 点绘制器: 官方调试用的小球 (挂 Material 上色, 自带回收池), 两个是为了大小不同
        self.wp_drawer = eng.make_point_drawer(scale=5.0 * WP_POINT_DIAM_M)
        self.look_drawer = eng.make_point_drawer(scale=5.0 * LOOK_POINT_DIAM_M)
        # 两个屏内示波器的区域 (比例坐标): 放在屏幕左侧, 避开 MetaDrive 自带的 HUD 文本框
        self._box_steer = (-0.97, 0.05, -0.45, 0.55)
        self._box_thr = (-0.97, -0.55, -0.45, -0.05)
        self._labels = {}
        self._build_frame()

    # ---------------------------------------------------------------- 内部
    def _p2(self, box, tx, val, full):
        """(时间分数 0~1, 值) -> 屏内比例坐标; tx=1 为现在 (右边界)"""
        x0, y0, x1, y1 = box
        v = min(max(val, -full), full)
        return x0 + tx * (x1 - x0), y0 + (0.5 + 0.5 * v / full) * (y1 - y0)

    def _2d(self, box, tx, val, full):
        x, y = self._p2(box, tx, val, full)
        return (x * self.ws, 0.0, y * self.hs)

    def _world(self, p, z=WORLD_POINT_Z):
        return (float(p[0]), float(p[1]), z)

    def _attach2d(self, np):
        """屏内曲线节点: 不设 CamMask (与 MetaDrive 自己的 draw_line_2d 一致), 每 tick 重建"""
        self._dyn.append(np)
        return np

    def _clear_dyn(self):
        # NodePath 的释放是 removeNode() (销毁节点本体) + remove() (从树上摘下来)
        for np in self._dyn:
            np.removeNode()
            np.remove()
        self._dyn = []

    def _text(self, box, dx, dy, s, size=0.032, static=True, corner="tl"):
        """在示波器框内放文字: corner 选锚角, (dx, dy) 为向内偏移 (比例坐标)"""
        tn = self._TextNode("lbl")
        tn.setText(s)
        right = corner in ("tr", "br")
        tn.setAlign(self._TextNode.ARight if right else self._TextNode.ALeft)
        np = self.aspect.attachNewNode(tn)
        x0, y0, x1, y1 = box
        px = (x1 - dx) if right else (x0 + dx)
        py = (y1 - dy) if corner in ("tl", "tr") else (y0 + dy)
        np.setPos(px * self.ws, 0.0, py * self.hs)
        np.setScale(size)
        np.setColor(1, 1, 1, 1)
        if static:
            self._static.append(np)
        else:
            self._dyn.append(np)
        return np

    def _build_frame(self):
        """静态: 两个框 + 零线 + 时间格线 (2 s 一根) + 标签"""
        for box, name, full in ((self._box_steer, "STEER", SCOPE_STEER_FULL),
                                (self._box_thr, "THROTTLE", SCOPE_THR_FULL)):
            seg = self._LineSegs()
            seg.setThickness(1.5)
            seg.setColor(0.75, 0.75, 0.75, 0.9)
            seg.moveTo(*self._2d(box, 0.0, -full, full))
            for (tx, val) in [(0.0, -full), (1.0, -full), (1.0, full), (0.0, full), (0.0, -full)]:
                seg.drawTo(*self._2d(box, tx, val, full))
            self._static.append(self.aspect.attachNewNode(seg.create(False)))
            # 零线与时间格线
            seg2 = self._LineSegs()
            seg2.setThickness(1.0)
            seg2.setColor(0.5, 0.5, 0.5, 0.6)
            seg2.moveTo(*self._2d(box, 0.0, 0.0, full))
            seg2.drawTo(*self._2d(box, 1.0, 0.0, full))
            for k in range(1, 5):
                tx = 1.0 - k / 5.0
                seg2.moveTo(*self._2d(box, tx, -full, full))
                seg2.drawTo(*self._2d(box, tx, full, full))
            self._static.append(self.aspect.attachNewNode(seg2.create(False)))
            self._text(box, 0.01, 0.03, f"{name} (last {self.win_s:.0f} s, +-{full:g})", corner="tl")
            self._labels[name] = self._text(box, 0.01, 0.03, "-", size=0.032, corner="tr")
            self._text(box, 0.01, 0.012, "t: -{:.0f}s ... now".format(self.win_s),
                       size=0.024, corner="bl")

    # ---------------------------------------------------------------- 对外
    def draw_waypoints(self, wp_world, look_xy=None):
        """把当前喂给控制器的 10 个 waypoints 画成 10 个离散球 (不连线)。

        颜色由近到远变暗 (绿 → 暗绿), 一眼看出点序与时间远近; 前视点用更大的黄球。
        不画连线是故意的: waypoints 是 2 Hz 的 10 个采样点, 连成折线会被当成“路径”误读。
        """
        self.wp_drawer.reset()
        self.look_drawer.reset()
        n = max(len(wp_world) - 1, 1)
        pts, cols = [], []
        for k, p in enumerate(wp_world):
            f = k / n
            pts.append(self._world(p))
            cols.append((0.10, 0.95 - 0.55 * f, 0.30, 1.0))
        self.wp_drawer.draw_points(pts, cols)
        if look_xy is not None:
            self.look_drawer.draw_points([self._world(look_xy, WORLD_POINT_Z + 0.15)],
                                         [(1.0, 0.85, 0.10, 1.0)])

    def draw_scopes(self, hist, now_t):
        """hist: 可迭代 (t, steer, throttle); 窗口外的样本自动丢弃"""
        self._clear_dyn()
        rows = [r for r in hist if now_t - r[0] <= self.win_s]
        if len(rows) < 2:
            return
        t_arr = np.array([r[0] for r in rows], dtype=float)
        steer = np.array([r[1] for r in rows], dtype=float)
        thr = np.array([r[2] for r in rows], dtype=float)
        tx = 1.0 - (now_t - t_arr) / self.win_s
        for box, vals, full, col in ((self._box_steer, steer, SCOPE_STEER_FULL, (0.15, 0.85, 1.0)),
                                     (self._box_thr, thr, SCOPE_THR_FULL, None)):
            seg = self._LineSegs()
            seg.setThickness(3.0)
            if col is not None:                        # steer: 单色
                seg.setColor(*col, 1.0)
                seg.moveTo(*self._2d(box, tx[0], vals[0], full))
                for a, b in zip(tx[1:], vals[1:]):
                    seg.drawTo(*self._2d(box, a, b, full))
            else:                                      # throttle: 正绿负红
                for a, b, c, d in zip(tx[:-1], vals[:-1], tx[1:], vals[1:]):
                    seg.setColor(*(0.2, 0.95, 0.3) if d >= 0 else (1.0, 0.3, 0.2), 1.0)
                    seg.moveTo(*self._2d(box, a, b, full))
                    seg.drawTo(*self._2d(box, c, d, full))
            self._attach2d(self.aspect.attachNewNode(seg.create(False)))
        self._labels["STEER"].node().setText(f"now {steer[-1]:+.3f} rad")
        self._labels["THROTTLE"].node().setText(f"now {thr[-1]:+.3f} (acc/brake)")

    def close(self):
        # 标签本身也在 _static 里 (由 _text 登记), 不单独销毁, 避免重复释放
        for np in self._dyn + self._static:
            np.removeNode()
        self._dyn = []
        self._static = []
        self._labels = {}
        for drawer in (self.wp_drawer, self.look_drawer):
            drawer.reset()
            drawer.removeNode()


def run(quick: bool = False, render: bool = False, spacing: float = 1.0,
        out_dir=None, verbose: bool = True, overlay=True, scope_window=SCOPE_WINDOW_S,
        render_fps=None, snap=()):
    from simple_track import LapCounter

    out_dir = Path(out_dir or (config.OUT_DIR / "validate_controller"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_x, ref_y, closed = tu.extract_lap_reference(config.OUT_DIR / "trajectory_expert.json")
    ref = ReferencePath(ref_x, ref_y, spacing=spacing)
    lat_tracker = tu.LateralErrorTracker(ref_x, ref_y)

    best = config.OUT_DIR / "best_geo_params.json"
    params = json.loads(best.read_text(encoding="utf-8")) if best.exists() else None
    ctrl = GeometricController(params)

    env = build_render_env(fps=render_fps) if render else tu.build_headless_env()
    rows, wp_ego_all, wp_world_all, s_all, sk_all, v_all = [], [], [], [], [], []
    # 滚动窗口: 仅用于屏内曲线绘制, 与落盘/判据数据无关
    scope_hist = deque(maxlen=max(2, int(round(scope_window / DT))))
    overlay_np = None
    wall_start = time.time()
    try:
        env.reset(seed=0)
        ctrl.reset()
        ref.reset()
        lat_tracker.reset()
        origin = np.array(env.agent.position[:2], dtype=float)
        lap_counter = LapCounter(origin)
        sim_t, laps, aborted = 0.0, 0, None
        max_sim_t = config.QUICK_SECONDS if quick else 1500.0
        low_speed_since = None
        snap_left = sorted(float(x) for x in snap)
        if verbose:
            print(f"[validate] 开始 {'quick' if quick else '整圈'} 10Hz, "
                  f"render={render}, 参考线长 {ref.L:.0f} m, 闭环={closed}", flush=True)

        while True:
            v = env.agent
            ego_xy = np.array(v.position[:2], dtype=float)
            heading = float(v.heading_theta)
            v_ms = float(v.speed_km_h) / 3.6

            wp_ego, wp_world, s_car, s_k = make_wp(ref, ego_xy, heading, v_ms)
            action = ctrl.compute(wp_ego, float(v.speed_km_h), DT)
            if render and overlay_np is None and overlay:
                overlay_np = SceneOverlay(env, scope_window=scope_window)
            if overlay_np is not None:
                scope_hist.append((sim_t, float(action[0]), float(action[1])))

            # 记录 = 本 tick 生成 waypoints 所用的位姿 (t = i*DT, 与方案 §1.1 对齐)
            offroad, collision = tu.vehicle_status(env, v)
            d0 = float(np.linalg.norm(wp_ego[0]))
            rows.append(dict(
                t=round(sim_t, 3), x=float(ego_xy[0]), y=float(ego_xy[1]),
                heading=heading, speed_kmh=float(v.speed_km_h),
                steering=float(action[0]), throttle_brake=float(action[1]),
                lat_err=lat_tracker.signed_error(ego_xy[0], ego_xy[1]),
                s_car=round(s_car, 3), wp0_dist_m=round(d0, 4),
                wp_phase_s=round(d0 / max(v_ms, 1e-6), 4),
                offroad=bool(offroad), collision=bool(collision),
            ))
            wp_ego_all.append(wp_ego)
            wp_world_all.append(wp_world)
            s_all.append(s_car)
            sk_all.append(s_k)
            v_all.append(v_ms)

            env.step(action)
            sim_t += DT
            v = env.agent

            if overlay_np is not None:
                # 画当前 tick 的 waypoints 与实际前视点 (用生成时刻的位姿, 与 ctrl.last_info 一致)
                try:
                    look_xy = None
                    if ctrl.last_info:
                        pts, arc = path_from_waypoints(wp_ego)
                        tgt = point_at_arc(pts, arc, float(ctrl.last_info["lookahead"]))
                        look_xy = ego_to_world(
                            np.array([[float(tgt[0]), float(tgt[1])]], dtype=float), ego_xy, heading)[0]
                    overlay_np.draw_waypoints(wp_world, look_xy)
                    overlay_np.draw_scopes(scope_hist, sim_t)
                except Exception as exc:
                    print(f"[validate] 叠加层绘制异常, 已关闭叠加层 ({type(exc).__name__}: {exc})", flush=True)
                    overlay_np.close()
                    overlay_np = None

            if render:
                env.render({
                    "t": f"{sim_t:6.1f} s",
                    "speed": f"{v.speed_km_h:5.1f} km/h",
                    "lat_err": f"{rows[-1]['lat_err']:+.3f} m",
                    "steer / throttle": f"{rows[-1]['steering']:+.3f} / {rows[-1]['throttle_brake']:+.3f}",
                    "lookahead": f"{ctrl.last_info.get('lookahead', float('nan')):5.2f} m",
                    "wp0 / v*0.5": f"{rows[-1]['wp_phase_s']:.3f} s (expect 0.500)",
                    "laps": f"{laps}",
                })
                if snap_left and sim_t >= snap_left[0]:
                    p = snap_window(env, out_dir, snap_left[0])
                    snap_left.pop(0)
                    if verbose:
                        print(f"[validate] snap t={sim_t:.1f}s -> {p}", flush=True)

            laps = lap_counter.update(v.position)
            if laps >= 1 and not quick:
                break
            if sim_t >= max_sim_t:
                aborted = "timeout"
                break
            if collision:
                aborted = "collision"
                break
            if sim_t > 10.0 and v.speed_km_h < 2.0:
                low_speed_since = sim_t if low_speed_since is None else low_speed_since
                if sim_t - low_speed_since > 5.0:
                    aborted = "stuck"
                    break
            else:
                low_speed_since = None
            if render and not getattr(env, "running", True):
                aborted = "user_exit"
                break
            if verbose and len(rows) % 1000 == 0:
                print(f"[validate] t={sim_t:7.1f}s speed={v.speed_km_h:5.1f}km/h "
                      f"lat={rows[-1]['lat_err']:+5.2f}m laps={laps}", flush=True)

        metrics = _summarize(rows, wp_ego_all, wp_world_all, s_all, sk_all, v_all,
                             ref, laps, quick, aborted, params,
                             off_limit=max(config.EVAL_OFFROAD_MAX_RATIO,
                                           expert_offroad_baseline(
                                               config.OUT_DIR / "trajectory_expert.json",
                                               rows[-1]["t"] if rows else 0.0) + OFFROAD_TOL))
        metrics["wall_time_s"] = round(time.time() - wall_start, 1)
        metrics["render"] = bool(render)
        metrics["ref_length_m"] = round(ref.L, 1)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        _save_csv(out_dir / "drive_log.csv", rows)
        _plot(out_dir, rows, ref_x, ref_y, metrics)
        _plot_phase(out_dir, rows, metrics)
        if verbose:
            print(f"[validate] 完成: laps={laps} aborted={aborted} "
                  f"mean|lat|={metrics['lat_abs_mean_m']:.3f}m "
                  f"T1_rel_max={metrics['T1_phase_rel_err_max']:.4f} "
                  f"→ {out_dir}", flush=True)
            for name, c in metrics["checks"].items():
                print(f"[validate]   {name}: value={c['value']} thr={c['threshold']} "
                      f"{'PASS' if c['ok'] else 'FAIL'}", flush=True)
            print(f"[validate] 总体 {'PASS' if metrics['passed'] else 'FAIL'} "
                  f"耗时 {metrics['wall_time_s']}s", flush=True)
        return metrics, out_dir
    finally:
        if overlay_np is not None:
            overlay_np.close()
        env.close()


def _check(name, value, threshold, ok, desc):
    return name, dict(value=round(float(value), 6) if np.isfinite(value) else None,
                      threshold=threshold, ok=bool(ok), desc=desc)


def _summarize(rows, wp_ego_all, wp_world_all, s_all, sk_all, v_all, ref,
               laps, quick, aborted, params, off_limit=None):
    """§5 判据: T1~T5 时序 + 跟随性能. 时序断均排除前 WARMUP_TICKS 个起步瞬态."""
    lat = np.array([r["lat_err"] for r in rows])
    speed = np.array([r["speed_kmh"] for r in rows])
    steer = np.array([r["steering"] for r in rows])
    wp_ego = np.asarray(wp_ego_all)[WARMUP_TICKS:]   # [N,10,2]
    v_ms = np.asarray(v_all)[WARMUP_TICKS:]
    s_all = np.asarray(s_all)
    n = len(rows)
    off_limit = config.EVAL_OFFROAD_MAX_RATIO if off_limit is None else off_limit

    # T1: 首点距离 = v·0.5s (连续滑动的直接证据; 若对齐 0.5 网格会锯齿到 0.1~0.5)
    exp0 = v_ms * WP_DT
    valid = exp0 > MIN_EXPECT_SPACING_M
    t1 = float(np.max(np.abs(np.linalg.norm(wp_ego[:, 0], axis=1)[valid] - exp0[valid])
                       / exp0[valid])) if valid.any() else 0.0
    # T2: 相邻点间距 = v·0.5s
    seg = np.linalg.norm(np.diff(wp_ego, axis=1), axis=2)      # [N,9]
    exp = np.repeat(exp0[:, None], 9, axis=1)
    ok2 = exp > MIN_EXPECT_SPACING_M
    t2 = float(np.median(np.abs(seg - exp)[ok2] / exp[ok2])) if ok2.any() else 0.0
    # T3: 里程推进 = v·0.1s, 且单调 (在完整 s_car 序列上差分后再截窗)
    ds_full = np.diff(s_all)
    ds_full[ds_full < -ref.L / 2] += ref.L                        # 回绕解卷
    ds = ds_full[WARMUP_TICKS - 1:]
    exp3 = v_ms * DT
    v3 = exp3 > 0.5
    t3 = float(np.max(np.abs(ds[v3] - exp3[v3]) / exp3[v3])) if v3.any() else 0.0
    t3_mono = float(np.min(ds))
    # T4a: ego→world 可逆 (用未截窗的原数组, 跳过起步 tick)
    rt = []
    for i in range(WARMUP_TICKS, n):
        rt.append(np.max(np.abs(ego_to_world(np.asarray(wp_ego_all[i]),
                                             (rows[i]["x"], rows[i]["y"]),
                                             rows[i]["heading"]) - np.asarray(wp_world_all[i]))))
    t4a = float(np.max(rt)) if rt else 0.0
    # T4b: 车在 τ_k 的实际位置 vs 当时发出的 wp_world[k] 的横向偏差
    lat_k = []
    for i in range(WARMUP_TICKS, n):
        for k in range(WP_N):
            j = i + int(round((k + 1) * WP_DT / DT))          # 5·(k+1) 个 tick 后
            if j >= n:
                continue
            th = float(ref.tangent_at(np.asarray(sk_all[i])[k])[0])
            ex = rows[j]["x"] - wp_world_all[i][k][0]
            ey = rows[j]["y"] - wp_world_all[i][k][1]
            lat_k.append(abs(-ex * np.sin(th) + ey * np.cos(th)))
    lat_k = np.asarray(lat_k) if lat_k else np.zeros(1)
    t4b = float(np.max(lat_k))
    # T5: 首点必须在前方, 且横向位置与参考线局部曲率一致
    #     (弯道上路径本身会弯, y 应为 ½κd²; 若把 y 轴符号写反则残差翻倍)
    from traj_utils import wrap_to_pi

    t5x = float(np.min(wp_ego[:, 0, 0]))
    th0 = ref.tangent_at(s_all[WARMUP_TICKS:])
    th1 = ref.tangent_at(s_all[WARMUP_TICKS:] + 10.0)
    kappa = wrap_to_pi(th1 - th0) / 10.0
    d0 = v_ms * WP_DT
    y_pred = 0.5 * kappa * d0 ** 2
    t5 = float(np.max(np.abs(wp_ego[:, 0, 1] - y_pred)))

    checks = dict([
        _check("T1_wp0_dist_eq_v_half", t1, f"<= {T1_REL_MAX}", t1 <= T1_REL_MAX,
               "首点距离相对误差 (窗口未随 t 滑动则爆表)"),
        _check("T2_spacing_eq_v_half", t2, f"<= {T2_REL_MED_MAX}", t2 <= T2_REL_MED_MAX,
               "相邻点间距中位相对误差"),
        _check("T3_s_car_step", t3, f"<= {T3_REL_MAX}", t3 <= T3_REL_MAX,
               "里程推进步长相对误差"),
        _check("T3_s_car_monotonic", t3_mono, f">= {T3_MONO_MIN_M}", t3_mono >= T3_MONO_MIN_M,
               "里程不回退 (投影量化/回绕出错则为负)"),
        _check("T4a_ego_roundtrip_m", t4a, f"<= {T4A_MAX_M}", t4a <= T4A_MAX_M,
               "ego↔world 可逆性"),
        _check("T4b_future_consistency_m", t4b, f"<= {T4B_MAX_M}", t4b <= T4B_MAX_M,
               "车实际到达位置 vs 当时承诺的 waypoints (坐标系约定写反则爆表)"),
        _check("T5_wp0_x_positive", t5x, "> 0", t5x > 0, "首点必须在前方"),
        _check("T5_wp0_y_curv_consistency", t5, f"<= {T5_Y_RES_MAX}", t5 <= T5_Y_RES_MAX,
               "首点横向 vs 局部曲率预测 ½κd² (y 轴写反则残差翻倍)"),
        _check("perf_lat_mean_m", float(np.mean(np.abs(lat))), f"<= {LAT_MEAN_MAX}",
               float(np.mean(np.abs(lat))) <= LAT_MEAN_MAX, "平均横向误差"),
        _check("perf_lat_p95_m", float(np.percentile(np.abs(lat), 95)), f"<= {LAT_P95_MAX}",
               float(np.percentile(np.abs(lat), 95)) <= LAT_P95_MAX, "p95 横向误差"),
        _check("perf_lat_max_m", float(np.max(np.abs(lat))), f"<= {LAT_MAX_MAX}",
               float(np.max(np.abs(lat))) <= LAT_MAX_MAX, "最大横向误差"),
        _check("perf_completed", float(laps >= 1 or quick), ">= 1", laps >= 1 or bool(quick),
               "完圈 (quick 只要未中止)"),
        _check("perf_collision_frames", float(sum(r["collision"] for r in rows)), "== 0",
               sum(r["collision"] for r in rows) == 0, "碰撞帧"),
        _check("perf_offroad_ratio", float(np.mean([r["offroad"] for r in rows])),
               f"<= {off_limit:.4f}",
               float(np.mean([r["offroad"] for r in rows])) <= off_limit,
               "出界帧占比 (阈值含专家同窗基线误报容差)"),
        _check("perf_speed_mean_kmh", float(np.mean(speed)), f">= {SPEED_MEAN_MIN}",
               float(np.mean(speed)) >= SPEED_MEAN_MIN, "均速"),
        _check("perf_speed_min_kmh", float(np.min(speed)), f">= {SPEED_MIN_MIN}",
               float(np.min(speed)) >= SPEED_MIN_MIN, "最低速"),
        _check("perf_steer_sat_ratio", float(np.mean(np.abs(steer) >= 0.6)),
               f"<= {STEER_SAT_RATIO_MAX}", float(np.mean(np.abs(steer) >= 0.6)) <= STEER_SAT_RATIO_MAX,
               "转向顶限幅帧占比"),
    ])
    return dict(
        mode="validate_controller", quick=bool(quick), aborted=aborted,
        completed=bool(laps >= 1) if not quick else aborted is None,
        laps=int(laps), duration_s=round(rows[-1]["t"], 2) if n else 0.0,
        num_ticks=n, params=params or {},
        lat_abs_mean_m=float(np.mean(np.abs(lat))), lat_abs_p95_m=float(np.percentile(np.abs(lat), 95)),
        lat_abs_max_m=float(np.max(np.abs(lat))), speed_mean_kmh=float(np.mean(speed)),
        speed_min_kmh=float(np.min(speed)), steer_abs_mean=float(np.mean(np.abs(steer))),
        T1_phase_rel_err_max=t1, T2_spacing_rel_med=t2, T3_step_rel_max=t3,
        T4a_roundtrip_max_m=t4a, T4b_future_lat_max_m=t4b, T4b_future_lat_p95_m=float(np.percentile(lat_k, 95)),
        offroad_limit_used=round(float(off_limit), 5), warmup_ticks=WARMUP_TICKS,
        passed=all(c["ok"] for c in checks.values()),
        checks=checks,
    )


def _save_csv(path, rows):
    import csv

    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot(out_dir, rows, ref_x, ref_y, metrics):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    step = max(1, len(ref_x) // 4000)
    ts = [r["t"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax = axes[0][0]
    ax.plot(ref_x[::step], ref_y[::step], "b-", lw=0.8, alpha=0.6, label="reference")
    ax.plot([r["x"] for r in rows], [r["y"] for r in rows], "r-", lw=0.9, label="driven")
    ax.set_aspect("equal")
    ax.set_title("Trajectory vs reference (no offset)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(ts, [r["lat_err"] for r in rows], lw=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"Lateral error (mean|.|={metrics['lat_abs_mean_m']:.3f} m)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("m")
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(ts, [r["speed_kmh"] for r in rows], lw=0.7, label="speed")
    ax.axhline(config.TARGET_SPEED_KMH, color="g", ls="--", lw=0.8, label="target")
    ax.set_title("Speed")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("km/h")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(ts, [r["steering"] for r in rows], lw=0.6, label="steering")
    ax.plot(ts, [r["throttle_brake"] for r in rows], lw=0.6, alpha=0.7, label="throttle/brake")
    ax.set_title("Control outputs (10 Hz)")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Controller validation: passed={metrics['passed']} "
                 f"completed={metrics['completed']} ticks={metrics['num_ticks']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "plots.png", dpi=130)
    plt.close(fig)


def _plot_phase(out_dir, rows, metrics):
    """D2 的可视化证据: 相位必须恒为 0.5s, 对齐网格会呈 0.1~0.5 锯齿."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ts = np.array([r["t"] for r in rows])
    phase = np.array([r["wp_phase_s"] for r in rows])
    dist = np.array([r["wp0_dist_m"] for r in rows])
    exp = np.array([r["speed_kmh"] / 3.6 * WP_DT for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax = axes[0]
    ax.plot(ts, dist, lw=0.7, label="actual |wp_ego[0]|")
    ax.plot(ts, exp, lw=0.7, ls="--", label="v * 0.5s")
    ax.set_title(f"T1: first waypoint distance (rel err max={metrics['T1_phase_rel_err_max']:.4f})")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("m")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ts, phase, lw=0.7)
    ax.axhline(WP_DT, color="g", ls="--", lw=0.8, label="expected 0.5 s")
    ax.axhline(0.1, color="r", ls=":", lw=0.8, label="grid-snapped bug would sawtooth 0.1~0.5")
    ax.set_ylim(0, 0.6)
    ax.set_title("Effective phase of first sample (must be flat 0.5 s)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "phase_check.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="只跑前 90s")
    parser.add_argument("--render", action="store_true", help="开窗口目视 (屏显英文)")
    parser.add_argument("--spacing", type=float, default=1.0, help="参考线重采样间距 m")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-overlay", action="store_true",
                        help="--render 下不画 waypoints / steer-throttle 曲线")
    parser.add_argument("--scope-window", type=float, default=SCOPE_WINDOW_S,
                        help="steer/throttle 曲线回看时长 s")
    parser.add_argument("--render-fps", type=float, default=None,
                        help="--render 下的渲染帧率上限, 传大值可快进 (默认约 50)")
    parser.add_argument("--snap", type=str, default="",
                        help="--render 下在这些仿真时刻存图, 逗号分隔, 如 200,270,340")
    args = parser.parse_args()
    snap = [float(x) for x in args.snap.split(",") if x.strip()]
    m, _ = run(quick=args.quick, render=args.render, spacing=args.spacing, out_dir=args.out,
               overlay=not args.no_overlay, scope_window=args.scope_window,
               render_fps=args.render_fps, snap=snap)
    raise SystemExit(0 if m["passed"] else 1)
