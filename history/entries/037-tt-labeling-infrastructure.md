# 037 · TT注入关联标签基础设施及生产复核

- 日期：2026-10-20
- 分支：feature/service-labeling-infrastructure；实现至c6e07e3
- 类型：Feature / Data / Docs / Test
- 影响域：src/service/labeling/、scripts/label_service_data.py、configs/service/tt_labels_a_v1.json、tests/test_service_label*、docs/research/service-20261019/

## 做了什么

实现来源绑定、窗口规则、版本化标签/划分、独立验收与稳定键diff；生成真实selected17独立侧表并新目录重放。85项指定测试通过，生产check和diff退出0；未训练或重建特征。

## 关键决策

- 延续035的Normal-only与整Case隔离，冻结Normal按时间4/1/1分配；Fault故障前仅测试。消化036后续标签基础设施部分，旧preparation全Unknown gate继续保留。
- 保留注入关联与协议正常语义，Unknown不转负类；窗口资格与资源/执行/日志有效性分开。
- 真实生产复核发现TRANSACTION的unknown_pause_policy，不能由缺暂停证据推断无需暂停；探查875个故障前负类撤销。这是生产证据纠正，不按预设数量改规则。实际计数及版本绑定在实现报告和本地manifest中，不在history复制实验表。
- 输出独立不可覆盖包；标签/划分语义hash与生成时间/路径分开，逐行重放diff为0。临时探查不再是正式标签入口。

## 坑 / 已知问题

- 最早Normal54条超时不等于54窗；重叠按窗口和全25服务排除。失败汇总无定位证据仅作警示，HTTP200不能证明业务成功。
- 三KILL瞬时操作和三数据库目标均无第一版应用窗口正类，不能删除事件或扩展持续/传播区间。
- 本地原始来源、历史绝对路径及大产物不进Git；远端仅源码不能直接取得完整生产资产。沙箱bwrap初始化失败，本轮本地命令通过授权require_escalated重试。

## 遗留 TODO

1. 依据版本化标签与模态mask开展后续模型接口；本轮不确定模型、阈值或训练方案。
2. 更广泛的KILL持续状态、数据库业务影响及B规则仍需独立证据，不把本轮关联标签当完整状态真值；已实现范围见下方v2收尾。
3. 最终审查可补公共loader合成fixture及完整六Case划分断言；现有85项检查不宣称覆盖未执行测试。

## v2收尾（2026-10-20）

- 用户批准扩展NetworkChaos参与对象与KILL双就绪缺失观察；来源及采集器30秒freshness约定均绑定hash。离开双缺失仅表示至少一种ready状态返回，不证明业务恢复；逐对象Remove回执缺失的限制保留。
- 主评价固定为独立Normal test负类与Fault已知正类；Fault注入前负类保留在标签表，通过独立evaluation清单排除，避免修改标签语义来调整评价比例。Unknown不转正常，4/1/1整Case划分不变。
- schema 2持久化评价清单并重算校验，模型主评价使用join_evaluation_scores，缺评分报错；旧schema 1仍可读取，历史包重放须匹配历史规则源码。
- v2六个相关测试文件共75项通过，正式包原始来源复验通过，重复构建稳定键diff为0；与首版85项为不同检查集合，不累加宣称。
- 正式接入、覆盖和证据限制见docs/research/service-20261019/labeling-v2-implementation-20261020.md；未来补采必须显式版本接入，尚未实现自动增量加载。本轮未训练、未改写原始数据。
