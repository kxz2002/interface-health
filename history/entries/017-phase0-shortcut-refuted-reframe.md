# 017 · Phase 0 诊断证伪 per-endpoint shortcut，转向 "Reliability ≠ Observability"

- **日期**: 2026-07-23
- **PR**: N/A（分析+方向决策，未改生产代码）· **Commit**: 待定（本 entry 随 C 的实现一起提交）
- **类型**: Docs
- **影响域**: 研究方向/论文 framing, `artifacts/contract_v1/`（只读诊断）, `.aris/traces/novelty-check/`, 后续 `scripts/`（C 的实现）

## 做了什么

在 `feature/per-endpoint-shortcut` 分支上，对"per-endpoint shortcut"这个论文核心假设做了立项前的双轮 novelty-check（外部审稿人 gpt-5.6-sol）+ 一轮 Phase 0 数据级诊断 + 一次数据采集仓库（AnoMod fork）审计。结论：**原 shortcut 假设被数据证伪，方向转向 "Reliability is not Observability"**。

原假设：朴素多模态融合因共享 service 级特征，会把被注入故障的 endpoint 的高异常分"传染"给同 service 的无辜 endpoint，使 per-endpoint 检测退化成 service-level 检测。

Phase 0 用 `artifacts/contract_v1/eval_all.parquet`（29 case，13632 行）+ endpoint_raw2 pipeline 产物做只读诊断，脚本在 `/tmp/phase0_*.py`（一次性诊断，未入库）。novelty-check 全过程 trace 在 `.aris/traces/novelty-check/2026-07-23_run01/`。

## 关键决策（不在 commit 里）

- **放弃 shortcut / granularity-leakage 叙事**：Phase 0 Finding 2 抽掉了机制前提——HTTP 层故障（Lv_E，16 case）下共享 service 级特征几乎不动（service_metric 偏移 0.02σ、service_log 0.15σ，对比 Lv_S killpod 的 log 0.61σ/ep_red 1.23σ）。没有被污染的共享协变量可"广播"，无辜 sibling 本就会正确地显示为正常，shortcut 无触发条件。补 sibling 也点不着火。审稿人定性：这是"信息缺失/观测不变性"，不是"伪相关(spurious)"，继续叫 shortcut 属误导。
- **转向 "Reliability ≠ Observability" 作为论文脊柱**：一个模态可在 Normal 数据上低方差、稳定（看似"可靠"），却对目标故障完全不变（实际无用）；把"稳定"当"可信"的 reliability gate 会压制唯一对故障敏感的分支。这直接解释 entry 014 的 RG 门控坍缩负结果（坍缩到信任低方差 service 分支不是 bug，是该机制的必然）。审稿人金句："恒定的传感器方差最小，也最没用。"这是非平凡的 systems-ML 交互点，且白捡已有 RG 资产。
- **先做 C（现有数据验证），再决定 A/B**：C = 在现有 contract_v1 上搭 "service-only 检测器 vs endpoint-only 检测器" 对比，确认 service 分支对 Lv_E 近乎随机、endpoint 分支能分开 ABORT/REPLACE。这是新叙事的实证地基，成本约一天。地基立住再决定是否升级（A=用现有数据搭 short paper 骨架 / B=正经 re-collection 冲更高 venue）。用户明确不追求 top-tier，倾向 C→A。
- **re-collection 暂不做**：审稿人裁定——只为"单 service sibling 图"不值得（现有数据已能证 service 特征平+目标 client_error 飙）；核心 reframe 用现有数据即可证，re-collection 只对"定位 claim"和"统计功效"有价值，且门槛是正经多 service 复现实验（≥3 service、每 service 3-5 并发 endpoint、traffic-share 扫、正/负对照、≥5 seed、run 级复现、检测 vs 定位分开评），不是轻量改脚本。

## 坑 / 已知问题（Phase 0 实测发现）

- **模型视野里每 service 恰好 1 个 endpoint，没有同 service sibling**：`build_contract.py:130` 用 `how="inner"` join trace 侧（~34 endpoint，含内部调用）和客户端 health 侧（8 顶层 endpoint）。inner join 后只剩 8 个 endpoint、每 service 一个（负载生成器只打 8 个顶层调用所致）。所以"真凶 vs 同 service 无辜邻居"这个测试在当前建模数据上根本跑不起来——endpoint 与 service 身份在模型视野里完全混同（1:1）。
- **HTTP 故障对 service 级 metric/log 近乎隐形**（决定性）：Lv_E 逐维——cpu 0.098→0.102、memory 0.674→0.677、process_count 0.667→0.667、net error 全 0；log error_ratio 0.008→0.017（翻倍但绝对值极小）、template_diversity 反而降。根因（与采集审计一致）：中止/延迟 HTTP 不耗 CPU/内存；TT 应用日志是业务方法日志（Hibernate/method 事件），不记代理层 HTTPChaos abort。
- **故障必须分层，勿把 16 个 Lv_E 混为一谈**：ABORT/REPLACE（8 case）client_error_rate ~0→0.97-1.0，endpoint-local 强信号；DELAY（4 case）只有 latency 且弱/混乱（HTTPABORT 下 trace latency 反而从 2743 掉到 15，快速失败）；PATCH（4 case）语义损坏，响应仍可能是 HTTP 200，**可能连 endpoint RED 都看不见**——这支持"观测覆盖度"论点，但推翻"endpoint 特征必能看见 endpoint 故障"的简单说法。
- **`endpoint_service` 列不可信**：同一个逻辑 endpoint（如 order/refresh）在网关层和服务层各记一次 entry span，被记到两个 service（gateway 46 行 + order 21 行）。定义"同 service"只能用 path 前缀（`/orderservice/`），不能用这列。
- **采集审计结论（AnoMod fork）**：TT 侧 metric 用 cAdvisor/node-exporter/kube-state-metrics，天生只到 pod/container/node 维度，且 TT 服务未暴露 `/actuator/prometheus`，连 `http_requests_total` 都无源；log collector 是零解析的 `kubectl logs` 转发，应用日志本身不含 route。所以 per-endpoint 粒度缺失是**采集架构结构性约束**，非 preprocessing bug。措辞注意：不能写"重采不可消除"（Prometheus 原则上能打 route label、Spring Boot 能开 Actuator），正确说法是"该粒度不对称是 released telemetry / 标准 service-scoped instrumentation 的内在属性"。
- **SN（DeathStarBench SocialNetwork）数据源是未来的雷**：其 nginx error log 含路由信息、metric 查询本可 `by(route)`。目前 SN 不在 `configs/data/*.yaml` 消费列表，不影响；若未来扩展消费 SN，这条"无 per-endpoint 粒度"论点需对 SN 单独重新核查，不能套 TT 结论。

## C 实测结果（2026-07-23，`scripts/analyze_modality_observability.py` → `artifacts/modality_observability/report.md`）

三个特征子集(endpoint_only 10维 / service_only 8维 / all 18维)各训 Deep SVDD × 4 seed，去跨run混杂(负=同case baseline/recover)。三条硬结论，都比原假设更精确：

- **信号淹没(核心，直接给方法立论)**：ABORT/REPLACE 的信号在**单特征** `client_error_rate` 里近乎完美(oracle AUROC 0.993/1.000)，但朴素等权全向量 SVDD：endpoint_only=0.278/0.419、all(18维)=0.426/0.499——**降到随机甚至反相关**。原因：ABORT 快速失败让 latency/request_count 反而更接近 normal 中心，error_rate 那一维被 L2 距离平均掉。这是 "reliability≠observability" 的**特征级版本**，与 entry 014 RG 分支级坍缩同源。naive fusion(all)对 Lv_E 整体仅 0.536。
- **"service 全盲"要修正为"service metric 盲、service log 间接可感但不可定位"**：service_only 对 ABORT within-case=0.822，来源是 service_log(event_rate 0.667/template_diversity 0.628，调用方记录下游失败)，**不是** service_metric(cpu/内存/process 仅 0.54-0.61)。但 service 特征每 service 共享，**无法定位是哪个 endpoint**。与 Phase 0 的 0.02σ 不矛盾：那是 metric 单变量均值偏移，这是多变量可分性。
- **跨run混杂真实且污染现有协议**：`eval_all` 用跨run Normal(采 07-11)当负样本对抗故障case(采 07-04)。service_only 靠 cpu/内存绝对水平分"哪个run"：ABORT 0.822→0.922、DELAY 0.437→0.696、REPLACE 0.469→0.717(cross-run 虚高)。**entry 012/014 的 L0/L1/L2/RG 数字都建在这个被污染协议上，部分测的是 run 身份**。
- **故障分层**：ABORT/REPLACE 强 endpoint-local 信号(error_rate)；DELAY 弱(靠 request_count~0.65)；PATCH 对所有 RED 特征隐形(oracle~0.5)，语义损坏需响应体校验。

## 遗留 TODO

- ~~C（新叙事的实证地基）~~ **已完成，见上方"C 实测结果"**。地基立住：信号淹没现象确认，naive fusion 对 endpoint 故障失效而信号确实存在(oracle 0.99)——这就是方法要解决的 gap。
- **跨run混杂必须修(优先级高，先于方法实现)**：现有 eval 协议拿跨run Normal 当负样本，虚高且污染所有 baseline 数字。方法实验前需改成 within-case 或同run负样本协议，否则新方法的增益无法与真实 baseline 比较。
- **方法设计方向(C 已给依据)**：保留 per-feature/per-branch 偏离证据、不让稳定维度用等权距离淹没敏感维度(如 max/top-k 偏离聚合、over-deviation attention、per-feature reliability 加权)。ABORT 的 oracle 0.99 vs naive 0.43 就是目标 headline 实验。
- **A（C 之后，若地基立住）**：用现有数据搭 "Reliability ≠ Observability" short paper 骨架——干预式观测审计（哪些模态响应哪类故障 scope）+ RG 失效实证（复用 entry 014）+ 轻量可识别性论证。目标 workshop/short paper。
- **B（可选加码，成本高）**：正经 re-collection（多 service、traffic-share 扫、run 级复现），冲更高 venue。用户不追求 top-tier，B 优先级低。
- **A（C 之后，若地基立住）**：用现有数据搭 "Reliability ≠ Observability" short paper 骨架——干预式观测审计（哪些模态响应哪类故障 scope）+ RG 失效实证（复用 entry 014）+ 轻量可识别性论证。目标 workshop/short paper。
- **B（可选加码，成本高）**：正经 re-collection（多 service、traffic-share 扫、run 级复现），冲更高 venue。用户不追求 top-tier，B 优先级低。
- **数据表示修正（审稿人建议，未落地）**：别再拿 inner-join 当标准数据集（它在缺 client/trace 时丢了 endpoint-local 数据）。应保留 endpoint-local 特征、service 特征每 service 存一份、把广播放进模型 adapter 而非存进 ground truth。此项与 C 不冲突，C 可先在现有 contract 上做。
- **一次性诊断脚本 `/tmp/phase0_*.py` 未入库**：若 C 的实现需要复用其中逻辑（如按 path 前缀定义 service、按故障家族分层统计偏移），应固化进 `scripts/` 或 `src/` 而非依赖 /tmp。
