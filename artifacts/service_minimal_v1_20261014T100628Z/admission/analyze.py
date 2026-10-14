"""Read frozen artifacts only; derive admission diagnostics without fitting/scoring."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
INPUTS = [
    "data/events.parquet",
    "data/features.parquet",
    "data/index.parquet",
    "data/evidence.parquet",
    "run/scores.parquet",
    "evaluation/event_rankings.parquet",
    "evaluation/event_comparison.csv",
    "evaluation/s1_proxy_metrics.json",
    "evaluation/s1_independent_metrics.json",
    "data/quality_report.json",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


before = {name: digest(ROOT / name) for name in INPUTS}
events = pd.read_parquet(ROOT / INPUTS[0])
features = pd.read_parquet(ROOT / INPUTS[1])
index = pd.read_parquet(ROOT / INPUTS[2])
evidence = pd.read_parquet(ROOT / INPUTS[3])
scores = pd.read_parquet(ROOT / INPUTS[4])
rankings = pd.read_parquet(ROOT / INPUTS[5])
rows = []
ties = []
for event in events.itertuples():
    core = index[(index.case_id == event.case_id) & (index.phase == "inject_core")]
    target_ids = set(core[core.canonical_service_id == event.configured_target_service].sample_id)
    target = features[features.sample_id.isin(target_ids)]
    common = dict(
        event_id=event.event_id,
        canonical_service_id=event.configured_target_service,
        window_start_ms=int(target.window_start_ms.min()),
        window_end_ms=int(target.window_end_ms.max()),
        n_expected=len(target),
        n_valid=int(target.resource_valid.sum()),
        evidence_ids=";".join(event.effect_evidence_ids),
        evidence_source="data/evidence.parquet: evidence_id; source_path/source_locator retained",
        window_source="data/index.parquet: phase=inject_core; data/features.parquet: sample_id",
    )
    if not target.resource_valid.any():
        assert len(target) == 35 and target.identity_ambiguity.all()
        rows.append(
            dict(
                common,
                category="不可恢复观测",
                finding="35/35目标核心窗IDENTITY_AMBIGUOUS；目标资源覆盖0；非有效输入模型错判",
                independent_validation="需真实活实例/Ready及正常重启或采集缺失对照",
                next_step="先补观测与评价；不借Killing清理实例",
            )
        )
    else:
        rows.append(
            dict(
                common,
                category="候选校准并列",
                finding="目标事件分数全8臂0.96饱和；R3较弱S2排名来自并列增多，非证明局部CPU辨识失败",
                independent_validation="预先冻结新增确认事件；独立跨Service校准验证",
                next_step="保留本次基线，不以Fault raw分数破并列",
            )
        )
    for group, arm in rankings[rankings.event_id == event.event_id].groupby(
        ["feature_group", "method"]
    ):
        tr = arm[arm.canonical_service_id == event.configured_target_service].iloc[0]
        count = (
            int(((arm.event_score - tr.event_score).abs() <= 1e-12).sum())
            if pd.notna(tr.event_score)
            else 0
        )
        ss = scores[
            (scores.case_id == event.case_id)
            & (scores.sample_id.isin(target_ids))
            & (scores.feature_group == group[0])
            & (scores.method == group[1])
        ]
        ties.append(
            dict(
                event_id=event.event_id,
                feature_group=group[0],
                method=group[1],
                target_score=tr.event_score,
                equal_scored_candidates_including_target=count,
                target_saturated_windows=int((ss.calibrated_score >= 0.96 - 1e-12).sum()),
                n_expected=len(target),
                n_valid=int(ss.anomaly_score.notna().sum()),
            )
        )
# Record categories explicitly even when no verified occurrence is established.
base = dict(
    event_id="all_3_events",
    canonical_service_id="all_25_services",
    window_start_ms="",
    window_end_ms="",
    n_expected="",
    n_valid="",
    evidence_ids="",
    evidence_source="evaluation/s1_independent_metrics.json; frozen evaluation tables",
    window_source="data/index.parquet",
)
rows.extend(
    [
        dict(
            base,
            category="标签错位/金标不足",
            finding="独立S1窗口标签0；仅3事件/1CPU/2目标/1Normal；Killing只证实例终止",
            independent_validation="语义匹配且独立正负窗口；新确认事件",
            next_step="独立AUROC/AP保持null",
        ),
        dict(
            base,
            category="Normal漂移",
            finding="尚未验证假设；饱和已知，但不能从高分或Normal_tail分位数证明漂移",
            independent_validation="独立健康实验及预定时间/服务对照",
            next_step="先补健康参考，不按Fault调校准",
        ),
        dict(
            base,
            category="有效输入下状态辨识失败",
            finding="未建立：CPU弱对照已高；KILL无有效目标输入；缺独立状态金标",
            independent_validation="有效观测与独立状态证据同时具备",
            next_step="不将R3检索较弱等同新模型必要性",
        ),
        dict(
            base,
            category="数据错误",
            finding="T5未证实新数据错误；CPU first_effect为跨注入起点counter差分区间起点；不是精确生效时刻，保留区间歧义",
            independent_validation="已有来源定位与时间单位/区间定义复核",
            next_step="不据该first_effect计算在线延迟；不改冻结产物",
        ),
    ]
)
pd.DataFrame(rows).to_csv(OUT / "failure_modes.csv", index=False)
pd.DataFrame(ties).to_csv(OUT / "calibration_ties.csv", index=False)
assert len(ties) == 24
cpu = pd.DataFrame(ties).query("event_id == 'Lv_P_CPU_preserve_20260728T164711Z_em'")
assert cpu.equal_scored_candidates_including_target.tolist() == [8, 3, 2, 3, 12, 19, 17, 18]
assert before == {name: digest(ROOT / name) for name in INPUTS}
outputs = ["failure_modes.csv", "calibration_ties.csv", "analyze.py"]
manifest = dict(
    consumed_sha256=before,
    output_sha256={name: digest(OUT / name) for name in outputs},
    command="conda run -n interface python artifacts/service_minimal_v1_20261014T100628Z/admission/analyze.py",
    scope="Existing artifact analysis only; no raw parse, fit, new scores, model tests or evaluation overwrite",
    assertions="24 arm/event rows; both KILL target35/35 ambiguity; CPU tie counts; consumed hashes unchanged",
)
(OUT / "analysis_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
)
print(cpu.to_string(index=False))
