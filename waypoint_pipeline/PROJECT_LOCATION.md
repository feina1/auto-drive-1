# 项目定位：我们的做法在开源/论文版图里的位置

> 本文件回答三个问题：
> 1. 我们现在的想法，和公开论文/开源项目里的做法**差在哪**；
> 2. 这个领域从入门到前沿**一共分几个阶段**，每阶段的门槛是什么；
> 3. **本项目现在处于哪个阶段**（location），以及下一步最该补什么。
>
> 结论先行（详见第 7 节）：
> **本项目 = 单场景、单前视、冻结 DINOv3 的 expert-only 行为克隆基线；
> 处于 L1（端到端 BC）的"工程框架已完成、正式精度实验未开始"位置，记作 L1.5。
> 输出协议（3 s / 6 点 ego waypoints）与 UniAD/VAD 的规划协议、TransFuser 的
> waypoint+PID 范式一致；差距不在思路上，在数据规模、输入信息量与评测协议上。**

调研时间：2026-11-18。所有数字均来自论文原文/官方仓库页，链接见第 2 节。

---

## 1. 先说结论：三句话

1. **思路上我们没有走偏。** "图像 → 未来若干 waypoints → 几何控制器"是 2021 年以来
   端到端驾驶最主流的一条线（TransFuser → VAD → VADv2 → Hydra-MDP 都是这个接口），
   我们选的 3 s / 6 点 / ego 系正好是 nuScenes 规划协议的标准规格。
2. **差距在三个"量"上：** 数据量（我们 1 小时 vs 文献 16–8,192 小时）、
   输入信息量（单前视 vs 6 相机 360°+LiDAR+地图+导航命令）、
   评测可比性（自建赛道 vs CARLA / Bench2Drive / NAVSIM 标准协议）。
3. **但有一项我们做得比多数论文好：可复现性。** baseline-v1 用 sha256 + 93 项不变量
   冻结，每个版本一张诊断图 + 迭代日志。这套东西在公开论文里很少见，
   它决定了我们能做"可信的消融研究"，而不是"不可复现的调参"。

---

## 2. 文献与来源（已逐条核对）

### 2.1 综述 / 路线图（用来定义"完整阶段"）

| 文献 | 出处 | 用途 |
|---|---|---|
| End-to-end Autonomous Driving: Challenges and Frontiers | TPAMI 2024，[arXiv:2306.16927](https://arxiv.org/abs/2306.16927) | 覆盖 270+ 篇，给出动机、roadmap、方法与未来趋势；本文阶段划分的主要依据 |
| 大模型时代端到端自动驾驶的认知架构演进与展望 | 自动化学报（中文综述） | 模块化→端到端→大模型/VLA/世界模型的演进；评测从开环转闭环 |
| Post-Training in End-to-End Autonomous Driving（awesome 仓库） | GitHub | 汇总 Bench2Drive / NAVSIM v2 / WOD-E2E 等后训练评测协议 |

### 2.2 与我们"同一条路线"的方法（waypoints + 控制器）

| 方法 | 出处 | 关键规格（与我们对照用） |
|---|---|---|
| TransFuser | CVPR 2021 / TPAMI 2023，[arXiv:2205.15997](https://arxiv.org/abs/2205.15997) | 图像+LiDAR Transformer 融合；输出 **T=4 个未来 waypoints（ego 系）**，`w_t = w_{t-1} + Δw_t`；横向/纵向各一个 PID 转控制；辅助任务：深度、语义分割、HD map、车辆检测；CARLA 0.9.10 + Leaderboard/Longest6 |
| TCP | NeurIPS 2022 | Trajectory-guided Control，轨迹与控制的联合预测，简单强基线 |
| VAD | ICCV 2023，[arXiv:2303.12077](https://arxiv.org/abs/2303.12077) | 全矢量化场景表示；6 相机 360°、2 Hz 关键帧；输入**自车状态 + 高层导航命令（左/右/直行）**；MLP planning head 输出 `V_ego ∈ R^{T_f×2}`；评测 **1 s / 2 s / 3 s** |
| UniAD | CVPR 2023 Best Paper，[arXiv:2212.10156](https://arxiv.org/abs/2212.10156) | planning-oriented：感知-预测-规划联合优化；planner 出轨迹后用牛顿法做碰撞优化；nuScenes **3 s 时域**、1/2/3 s 评测 |
| VADv2 | [arXiv:2402.13243](https://arxiv.org/abs/2402.13243) | **概率**规划：输出轨迹分布而非单条；**T=6 个路点、间隔 0.5 s（≈3 s）**；规划词表约 4096 条；PID 转控制；CARLA / Bench2Drive / NAVSIM |
| Hydra-MDP | CVPR 2024 NAVSIM 竞赛第一，[arXiv:2406.06978](https://arxiv.org/abs/2406.06978) | 轨迹词表来自 nuPlan 70 万条轨迹 K-means（V8192）；每条 **40 个时间戳 = 4 s @ 10 Hz**；教师-学生多目标蒸馏；指标 PDMS |
| DiffusionDrive | CVPR 2025 | 截断扩散做多模态规划，实时性取向 |

### 2.3 评测协议（决定"能不能对外比较"）

| 基准 | 出处 | 协议要点 |
|---|---|---|
| nuScenes | CVPR 2020 | 1000 场景 × 20 s；6 相机；2 Hz 标注；约 75% 为简单直行；**只支持开环** |
| CARLA / Leaderboard 2.0 | CoRL 2017，[arXiv:1711.03938](https://arxiv.org/abs/1711.03938) | 闭环仿真；DS（驾驶分）/ RC（路线完成）/ IS（违规分）；2.0 为长路线 |
| NAVSIM | NeurIPS 2024，[arXiv:2406.15349](https://arxiv.org/abs/2406.15349) | 真实数据 + **4 s 非反应式 pseudo-closed-loop**；PDMS = NC×DAC×(5·EP+5·TTC+2·C)/12；CVPR 2024 竞赛 143 队 / 463 次提交；**明确结论：开环位移误差不能反映闭环性能，简单 TransFuser 可媲美大型 E2E 模型** |
| NAVSIM v2 | CoRL 2025 | EPDMS，扩展舒适性/合规性等多维指标 |
| Bench2Drive | NeurIPS 2024 D&B，[arXiv:2406.03877](https://arxiv.org/abs/2406.03877) | CARLA v2；**200 万帧、13,638 clips、44 场景、23 天气、12 城镇**；**220 条闭环路线**；DS / SR / Efficiency / Comfort 多能力评测；专家为 RL 世界模型 Think2Drive |
| Pseudo-Simulation | CoRL 2025，[arXiv:2506.04218](https://arxiv.org/abs/2506.04218) | 3DGS 合成新视角观测做伪闭环；与真实闭环相关性 R²=0.8，优于最好开环的 R²=0.7 |
| WOD-E2E | CVPR 2026 | Waymo 长尾开环基准，RFS 指标 |
| MetaDrive | TPAMI 2022，[GitHub](https://github.com/metadriverse/metadrive) | 程序化生成多样场景；**自带"训练/测试不同地图"的泛化基准**；ScenarioNet 可导入真实数据集 |

### 2.4 数据与方法论（解释"我们为什么会有天花板"）

| 文献 | 出处 | 结论（对我们的意义） |
|---|---|---|
| Codevilla et al., Exploring the Limitations of Behavior Cloning for Autonomous Driving | ICCV 2019，[arXiv:1904.08980](https://arxiv.org/abs/1904.08980) | BC 能在未见环境做复杂操作，但有 **dataset bias、过拟合、动态物体处理差、缺因果建模、训练不稳**；提出 coiltraine 基准 |
| Codevilla et al., On Offline Evaluation of Vision-based Driving Models | ECCV 2018 | **离线指标（转向 MSE 等）与在线驾驶性能相关性弱**；偶发大误差会被平均值掩盖 |
| Exploring Data Aggregation in Policy Learning for Vision-based Urban AD | CVPR 2020 | DAgger 式数据聚合显著优于纯 BC |
| Learning from All Vehicles（PARAD） | CVPR 2022 | 用**所有车辆**当专家，无需额外采集即可扩充数据分布 |
| Data Scaling Laws for End-to-End Autonomous Driving | CVPR 2025 Workshop，[arXiv:2504.04338](https://arxiv.org/abs/2504.04338) | 内部数据从 **16 小时到 8,192 小时**，开环与闭环同步评估数据量-性能关系 |
| DINOv3 | 2025-08，[arXiv:2508.10104](https://arxiv.org/abs/2508.10104) | 自监督视觉基础模型；Gram anchoring 解决密集特征退化；ViT-S/16 约 21M 参数；**高质量 dense feature，可直接冻结 + 线性探针**——我们用的就是这个范式 |
| SimLingo | CVPR 2025，[arXiv:2503.09594](https://arxiv.org/abs/2503.09594) | VLM 纯视觉，Bench2Drive SOTA，CARLA Challenge 2024 冠军；语言-动作对齐 |
| DriveLM / DriveVLM | ECCV 2024 / CoRL 2024 | 语言/视觉语言模型做驾驶推理，L5 代表 |
| IV 2024 对比研究（waypoint vs direct control） | IEEE IV 2024 | **waypoint 管线（TransFuser）比直接出控制量（CIL++）对换车/换平台更鲁棒**——为我们的接口选择提供背书 |
| openpilot | [GitHub](https://github.com/commaai/openpilot) | 开源量产 ADAS，工业侧参考（端到端感知 + 规划 + 控制，含实车安全栈） |

---

## 3. 这个领域的完整阶段（L0–L6）

> 说明：L0–L6 是**本文按"能力门槛 + 评测协议"归纳的分级**，不是论文里的官方术语。
> 目的是给自己定位，不是给别人的工作排名。

| 阶段 | 能力定义 | 代表工作 | 数据量级 | 评测门槛 | 进入门槛 |
|---|---|---|---|---|---|
| **L0** 模块化 / 规则 | 感知→预测→规划→控制 分模块；规则或在线优化求解 | Apollo、Autoware、Stanley、MPC | 无需学习或少量 | 实车/仿真里程、接管率 | 系统工程与标定 |
| **L1** 端到端 BC | 单网络 图像→动作/轨迹；专家数据监督；控制器兜底 | PilotNet、CIL、**TransFuser**、TCP | 10⁰–10² h | 开环 L2/MSE + CARLA 闭环 | 专家数据 + 闭环仿真 |
| **L2** 结构化中间表征 | BEV/矢量/占用 + 多任务（检测/分割/地图/预测/规划）联合 | NEAT、**UniAD**、**VAD** | 10²–10³ h（nuScenes 1000×20 s） | nuScenes 3 s 开环 + CARLA | 多相机 + 3D 标注 + 多任务训练 |
| **L3** 概率 / 多模态规划 | 轨迹词表、扩散、概率头；输出多模态分布 | **VADv2**、**Hydra-MDP**、DiffusionDrive | 10²–10³ h | **NAVSIM PDMS（4 s 非反应式）** | 大规模轨迹词表 + 指标工程 + 教师蒸馏 |
| **L4** 标准闭环基准 / 数据引擎 | 统一闭环协议 + 自动难例挖掘 + 多能力评分 | **Bench2Drive**、NAVSIM v2、Pseudo-Simulation | 10⁶ 帧级 / 10³ h | 220 条路线 DS/SR；3DGS 伪闭环 | 仿真工程 + 数据引擎 + RL 专家 |
| **L5** 基础模型 / VLA | 视觉-语言-动作统一；自然语言指令与解释 | DriveLM、DriveVLM、**SimLingo** | 10³–10⁴ h + 语言标注 | CARLA Leaderboard 2.0 / Bench2Drive SOTA | 大模型训练 + 语言数据 + 算力 |
| **L6** 世界模型 / 生成式仿真 | 生成未来观测与场景，在"想象"里闭环 | GAIA-1、DriveDreamer、Vista | 10⁴ h 视频 | 生成质量 + 下游闭环增益 | 视频生成 + 大规模算力 |

**每一阶段真正的门槛（不是模型结构，而是这三样）**：
1. **数据**——能不能拿到足够覆盖的长尾；
2. **评测**——有没有别人认的协议，否则数字无法比较；
3. **工程**——闭环能不能稳定跑完、能不能复现。

---

## 4. 我们项目的客观事实（核查自仓库，非估计）

| 项 | 值 | 来源 |
|---|---|---|
| 地图 | SimpleTrack 闭环，约 10.8 km | `config.py` |
| 场景 | 固定赛道、**无交通、无天气变化、无动态障碍** | `SimpleTrackEnv` |
| 圈数 / 速度 | 5 圈：40 / 48 / 56 / 64 / 72 km/h | `dataset_waypoints_5laps_meta.json` |
| 总时长 | **3,637.1 s ≈ 1.01 h** | 同上 `trajectory_metadata.duration_s` |
| 遥测 | 10 Hz，36,371 帧；0 碰撞、447 出界帧（原样保留） | 同上 |
| 图像 | 640×360、FOV 60°、**2 Hz**，7,274 张，约 268 MB | 同上 |
| 标签 | 6 点、3.0 s、2 Hz、ego 系（x 前 y 左），t+0.5…t+3.0 | 同上 |
| 标签校验 | `spacing_median_rel_err = 0.0001`；wp0 中位 5.556/6.667/7.778/8.889/10.0 m | 同上 |
| 划分 | 按圈：train 第 1–4 圈 30,894 帧 / val 第 5 圈 5,477 帧（遥测）；带图帧 **6,178 / 1,096** | 同上 |
| 输入 | 单帧 RGB + `speed/72` + `steering/0.6` | `TRAINING_PLAN.md` §2 |
| 模型 | DINOv3 ViT-S/16 **冻结** → [384] → MLP → 6×2；可训练约 0.17M | 同上 |
| 训练配方 | 加权 SmoothL1(β=1 m)+0.02 二阶平滑；AdamW 3e-4；Cosine 300 ep；batch 64 | 同上 §3 |
| 控制器 | GeometricController（Pure Pursuit + PID + 曲率限速），10 Hz | `controllers.py` |
| 控制器自证 | GT waypoints 喂入：70 km/h、平均横向 0.018 m、整圈 PASS | `Controller验证方案.md` |
| 闭环 | 自建，跑 SimpleTrack 整圈；v1 冒烟 20 s 上限、推理 6.24 ms/次 | `reports/v1/summary_metrics.json` |
| 可复现 | baseline-v2：93 项不变量 + 4 项 sha256，`baseline_check.py` | `BASELINE.md` |
| 进度 | 流水线、数据、控制器、诊断全部完成；**E1–E7 精度实验一个都没跑** | `README.md` §7 |

v1 报告（TinyCNN / 300 帧 / 2 epochs / 只验证链路）的读数：
ADE 38.76 m、FDE 59.79 m、预测平均长度 0.2 m vs GT 35.0 m、闭环 20 s 截断、出界 93.5%。
**这些数字不构成模型能力结论**，只证明管道通、诊断图能读出"预测坍缩"。

---

## 5. 逐项对照：我们 vs 文献主流

| 维度 | 本项目 baseline-v2 | 文献主流（L1–L3） | 差距性质 |
|---|---|---|---|
| 场景 | 固定赛道 10.8 km，无交通 | CARLA 12 城镇 / 23 天气 / 44 场景；nuScenes 真实城市 | **多样性差 2–3 个量级** |
| 数据量 | 1.0 h / 36,371 帧遥测 / 7,274 张图 | 16–8,192 h（缩放律）；TransFuser 210 GB；Bench2Drive 200 万帧 | **1–3 个量级** |
| 数据来源 | 规则专家，**无扰动、无 DAgger** | DAgger/数据聚合、RL 专家（Think2Drive）、全体车辆当专家（PARAD） | **方法论差异（有意为之）** |
| 输入视角 | 单前视 640×360 FOV60 | 6 相机 360°（UniAD/VAD）；图像+LiDAR（TransFuser） | 信息量 |
| 导航命令 | **无**（模型只能"沿路"） | VAD/UniAD 显式输入左/右/直行命令 | 信息量（关键） |
| 时序 | 单帧 | 多帧时序融合 | 信息量 |
| 输出规格 | 6 点 × 3 s @ 2 Hz ego 系 | UniAD/VAD 3 s；TransFuser 4 点；VADv2 6 点 0.5 s；Hydra-MDP 40×10 Hz=4 s | **基本对齐** |
| 模型 | DINOv3 ViT-S/16 **冻结** + 0.17M head | 端到端微调 + 多任务头（检测/分割/地图/预测/规划） | 容量与任务数 |
| 损失 | 单任务回归 + 平滑项 | 多任务 + 教师蒸馏（Hydra-MDP）+ RL 后训练 | 复杂度 |
| 闭环 | 自建 GeometricController，10.8 km 整圈 | PID（TransFuser）、牛顿法优化（UniAD）、PDMS 非反应式（NAVSIM） | **协议不可比** |
| 评测指标 | 自定义 ADE/FDE + 自建闭环（lat/offroad/collision） | CARLA DS/RC/IS、Bench2Drive DS/SR、NAVSIM PDMS | **不可对外比较** |
| 仿真平台 | MetaDrive（**本仓库自带 `metadrive-main/`**） | CARLA / nuScenes / nuPlan | 平台不同，且未用 MetaDrive 的泛化协议 |
| 可复现 | **sha256 冻结 + 93 项不变量 + 单图诊断 + 版本日志** | 多数给代码，不给数据哈希与不变量校验 | **优于多数** |

---

## 6. "我们的想法"与"文献想法"的六个区别

### 6.1 输出接口：我们和主流一致，这点不用改

"网络出未来 waypoints，再由几何/优化控制器转成油门刹车"是 TransFuser（2021）以来
贯穿 VAD（2023）、VADv2（2024）、Hydra-MDP（2024）的公共接口。
IV 2024 的对比研究还专门验证了 **waypoint 管线比直接出控制量对换平台更鲁棒**。
我们选 3 s / 6 点 / ego 系，正好落在 nuScenes 规划协议（3 s、1/2/3 s 评测）上。
**结论：接口层面我们不在"入门水平"，而是在主流规格上。**

### 6.2 数据哲学：我们是"干净下界"，不是"最终系统"

文献里几乎没有人只用一遍干净专家数据：
DAgger 式聚合（CVPR 2020）、全体车辆当专家（PARAD 2022）、RL 世界模型专家（Think2Drive）、
多天气多城镇（Bench2Drive）都是标配。Codevilla 2019 明确列出纯 BC 的五个病灶：
dataset bias、过拟合、动态物体、缺因果建模、训练不稳定。

但反过来看：**因为没有扰动、没有筛选，baseline-v2 是一个干净的下界**。
它的价值不是"性能高"，而是"任何后续改动都能归因"。这是刻意的设计选择，
代价是：**它注定无法代表最终系统能力**，72 km/h 闭环就是它的 OOD 测试。

### 6.3 输入信息量：我们缺"去哪"和"周围有什么"

这是最实质的差距，不是调参能补的：

- **缺导航命令**：VAD/UniAD 都把左/右/直行作为显式输入。我们没有，
  所以模型只能学"沿着这条路走"。换地图、换目的地、进路口必崩。
- **缺 360°**：6 相机环视（nuScenes/VAD）才能看到侧后方；单前视 FOV60 在弯道内侧有盲区。
- **缺历史帧**：单帧看不出运动趋势，急弯入口的转向意图只能靠图像猜。

### 6.4 表征与容量：我们是 linear probe，天花板是明确的

DINOv3 论文本身就是"冻结特征 + 线性探针"的范式（arXiv:2508.10104），
所以我们的做法在方法论上站得住。但要清楚代价：
DINOv3 是**通用**密集特征，不是驾驶专用；0.17M 的 head 无法补偿。
L2 以上的论文都是端到端微调 + 多任务头，骨干与任务一起学。
**我们省了算力，也锁死了上限。** 这不是错，是取舍。

### 6.5 规模：1 小时不在"数据缩放律"的可观测区间里

数据缩放研究（arXiv:2504.04338）的区间是 **16 到 8,192 小时**。
我们 1.0 小时连下限都不到。这意味着：

- 我们**不能**声称任何 scaling 结论；
- E1 的 ADE 好坏只能说明"这个配置在 1 小时数据上收敛到什么程度"；
- 想进入文献可比较的区间，至少要 16 小时量级（约 16 倍现有数据）。

### 6.6 评测：自建赛道 + 自定义 ADE，不能对外比较

文献反复警告两件事，我们都撞上了：

1. **开环指标不能外推闭环**（Codevilla ECCV 2018；NAVSIM 2024 摘要直接写
   "开环位移误差不能反映闭环性能"）。我们的"ADE < 3 m"只能当内部门槛。
2. **没有标准协议就没有可比性**。CARLA 用 DS/RC/IS，Bench2Drive 用 DS/SR，
   NAVSIM 用 PDMS（4 s 非反应式）。我们自建的 20 s/整圈闭环，
   哪怕跑得再好，也无法写进任何榜单或与论文对表。

我们做对的一点是：**同时出开环和闭环两张图**，这至少能在内部观察两者是否一致。

### 6.7 反过来：我们有一项文献里稀缺的东西

论文普遍只给代码，不给**数据哈希 + 不变量校验 + 逐版本诊断 + 迭代日志**。
baseline-v2 的 93 项不变量意味着：任何人拿到这份数据，都能确认"这是同一份数据"。
这让我们的定位可以不是"刷榜"，而是**"可复现的机制消融平台"**：

- E3 车速输入必要性（`--zero-prop`）
- E4 时域 3 s vs 5 s
- E5 图像密度 2 Hz vs 1 Hz
- 开环 ADE 与闭环横向误差的相关性（能不能复现"弱相关"这个文献结论）

这几类实验在 1 小时数据 + 强控制器的条件下**做得动、也说得清**，
而刷 CARLA 榜单需要的是几百小时数据和几十张 GPU。

---

## 7. 定位结论：L1.5

### 7.1 逐阶段排除

| 判断 | 理由 |
|---|---|
| **不是 L0** | 我们没有分模块的感知/预测栈，是单网络直接出轨迹 |
| **是 L1** | expert-only 行为克隆 + waypoints 输出 + 几何控制器兜底 + 开环/闭环双评测 |
| **不是 L2** | 无 BEV/矢量/占用表征、无多任务头、无 360°、无导航命令、无时序融合 |
| **不是 L3** | 确定性单峰回归，没有轨迹词表/扩散/概率分布 |
| **不是 L4** | 没有接入任何标准闭环基准，评测协议自建不可比 |
| **不是 L5/L6** | 无语言、无 VLA、无世界模型 |

### 7.2 为什么是 L1.5 而不是 L1

L1 的**工程框架**我们做得比典型 L1 论文更完整：

- 数据冻结（sha256 + 93 项不变量）——论文少见；
- 控制器单独自证（GT waypoints 整圈 PASS，横向 0.018 m）——论文少见；
- 版本化离线诊断图 + 迭代日志——论文少见；
- 开环 + 闭环双评测脚手架——论文常见，我们已具备。

但 L1 的**科学结论**是空的：

- E1 没跑（正式训练一次都没做）；
- v1 报告是 TinyCNN 300 帧 2 epochs 的链路冒烟，预测坍缩到原点；
- 唯一"闭环"是 20 s 截断 + 72 km/h 未见速度，证明不了任何精度。

**所以准确的说法是：L1 的框架完成度约 80%，L1 的科学结论完成度约 0%。**
打个比方：相当于一篇 L1 论文的 Method + Experiment Setup 章节写完了，
Results 那一节的表格还是空的。

### 7.3 一句话版本

> 我们现在站在 TransFuser/VAD 那条路的起点：
> 采集管线、控制接口、评测脚手架都搭好了，
> **还没跑出第一个能看的精度数字**。

---

## 8. 差距清单与优先级

### A. 不动 baseline-v2（只训练/评测，当前可直接做）

| 优先级 | 动作 | 目的 | 对应文档 |
|---|---|---|---|
| **P0** | E1：DINOv3 全量 30,894 帧、300 epochs、验证第 5 圈 | 从 L1.5 变成"L1 完成"，拿到第一个可信 ADE/FDE | TRAINING_PLAN §4.1 |
| **P0** | E6：五档速度闭环（40/48/56/64/72） | 得到"闭环 vs 速度"曲线，找 OOD 拐点 | §4.6 |
| **P1** | 开环-闭环一致性分析 | 把每帧 ADE 与闭环横向误差对齐，验证文献"弱相关"结论在我们这里是否成立 | 新增分析脚本 |
| **P1** | E3：`--zero-prop` 车速消融 | 验证"必须输入车速"这个设计假设 | §4.3 |
| **P2** | E4：3 s vs 5 s 时域消融 | 给"为什么选 3 s"提供实验依据 | §4.4 |
| **P2** | E5：2 Hz vs 1 Hz 图像密度 | 决定存储策略 | §4.5 |

### B. 需要另立 baseline-v3（**不改动 v2**，另起数据版本）

| 优先级 | 动作 | 补的是哪块短板 |
|---|---|---|
| P1 | 加历史帧（t-0.5 s） | 时序信息（README 已知限制第 1 条） |
| P1 | 加导航命令（左/右/直行 或 目标点） | "去哪"（VAD/UniAD 的必备输入） |
| P2 | 图像扰动 / 亮度 / 天气 | 域随机化（E7），对应 Bench2Drive 的 23 天气 |
| P2 | DAgger / 数据聚合 | distribution shift（Codevilla 2019 的主要病灶） |
| P3 | MetaDrive 多地图（训练/测试不同地图） | 泛化协议（MetaDrive 自带，我们没用） |

> 注意：B 类全部违反"这一轮不改数据"的约束，**只作为未来版本计划，不在当前执行**。

### C. 远期（换平台，不是本仓库能直接对接的）

| 动作 | 门槛 | 收益 |
|---|---|---|
| 接入 NAVSIM（4 s、PDMS） | nuPlan 数据 + 官方评测栈 | 最便宜的"可对外比较"入口 |
| 接入 CARLA Leaderboard 2.0 | CARLA + 长路线闭环工程 | DS/RC/IS 标准分 |
| 接入 Bench2Drive（220 路线） | CARLA v2 + 200 万帧级数据 | 多能力评测（DS/SR/Efficiency/Comfort） |
| 多相机 360° + 时序融合 | 改采集与模型 | 进入 L2 |
| 轨迹词表 / 扩散头 | 改输出头 + 教师蒸馏 | 进入 L3 |

---

## 9. 边界与风险（读数字前必看）

1. **固定赛道 ≠ 泛化。** MetaDrive 自带的泛化基准就是"训练/测试不同地图"，我们连这个协议都没进。
   任何"模型学会了开车"的说法都不成立，只能说"在这条赛道上拟合了专家"。
2. **1 小时数据低于缩放律区间下限（16 h）。** 不在可观测区间内，不能谈 scaling。
3. **72 km/h 是未见速度（OOD）。** v1 冒烟闭环 93.5% 出界是**预测坍缩**导致，
   不是速度泛化结论；E6 跑完之前不要引用这个数字。
4. **447 个出界帧说明专家本身不完美。** BC 会照学这些偏差（dataset bias），
   这是 v1 数据"如实保留"的必然代价，不是 bug。
5. **单前视 FOV60 有弯道内侧盲区。** 这是物理信息缺失，调 head 补不回来。
6. **无交通 → 不涉及交互。** L4 的"安全性"维度（碰撞率、TTC、舒适性）我们完全没测。
7. **开环 ADE 与闭环性能弱相关**（Codevilla 2018、NAVSIM 2024）。
   E1 的 "ADE < 3 m" 是内部门槛，不是能力证明。

---

## 10. 术语对照（读论文时用）

| 我们的说法 | 论文里的说法 |
|---|---|
| waypoints / 6 点轨迹 | future waypoints、trajectory、planning output |
| 开环 ADE / FDE | open-loop L2 error（1 s / 2 s / 3 s） |
| 闭环（自建） | closed-loop evaluation、CARLA Leaderboard DS |
| 几何控制器 | PID controller、pure pursuit、lateral/longitudinal control |
| expert-only BC | behavior cloning、imitation learning |
| 数据聚合（未来） | DAgger、data aggregation |
| 冻结特征 + 轻头 | frozen backbone、linear probing |
| 固定赛道 | single-scenario、in-distribution |
| 预测坍缩 | mode collapse / regression to the mean |
| 我们的 baseline-v2 | 文献里的 "expert dataset" + "reproducibility package" |

---

## 11. 参考链接汇总

- TPAMI 2024 综述：https://arxiv.org/abs/2306.16927
- TransFuser：https://arxiv.org/abs/2205.15997
- UniAD：https://arxiv.org/abs/2212.10156
- VAD：https://arxiv.org/abs/2303.12077
- VADv2：https://arxiv.org/abs/2402.13243
- Hydra-MDP：https://arxiv.org/abs/2406.06978
- NAVSIM：https://arxiv.org/abs/2406.15349
- Bench2Drive：https://arxiv.org/abs/2406.03877
- Pseudo-Simulation：https://arxiv.org/abs/2506.04218
- SimLingo：https://arxiv.org/abs/2503.09594
- DINOv3：https://arxiv.org/abs/2508.10104
- Data Scaling Laws for E2E AD：https://arxiv.org/abs/2504.04338
- Codevilla 2019（BC 局限）：https://arxiv.org/abs/1904.08980
- CARLA：https://arxiv.org/abs/1711.03938
- MetaDrive：https://github.com/metadriverse/metadrive （TPAMI 2022）
- openpilot：https://github.com/commaai/openpilot