"""per-case 宏平均指标：只对同时含正负类的 case 取均值。"""

from __future__ import annotations

import pandas as pd

from scripts.eval_baseline_v0 import compute_stratified_metrics


def _row(case, sample, score, pos):
    return {
        "sample_id": sample,
        "score": score,
        "y_true": int(pos),
        "is_endpoint_anomaly": bool(pos),
        "case_id": case,
        "endpoint_key": "ep1",
        "phase": "inject" if pos else "baseline",
        "anomaly_type": case,
        "anomaly_level": "Lv_E",
        "label_granularity": "endpoint",
    }


def test_per_case_macro_skips_single_class_cases():
    # caseA 完美可分（AUROC 1.0），caseB 全负（单类，必须被跳过）
    rows = [
        _row("caseA", "a1", 0.1, False),
        _row("caseA", "a2", 0.9, True),
        _row("caseB", "b1", 0.5, False),
        _row("caseB", "b2", 0.6, False),
    ]
    m = compute_stratified_metrics(pd.DataFrame(rows))
    assert m["per_case_auroc_macro"] == 1.0
    assert m["per_case_auprc_macro"] == 1.0
    assert m["n_cases_with_both_classes"] == 1


def test_per_case_macro_all_single_class_returns_none():
    # 两个 case 全部只有负样本：没有任何双类 case，宏平均无定义，必须为 None
    rows = [
        _row("caseB", "b1", 0.5, False),
        _row("caseB", "b2", 0.6, False),
        _row("caseD", "d1", 0.4, False),
        _row("caseD", "d2", 0.7, False),
    ]
    m = compute_stratified_metrics(pd.DataFrame(rows))
    assert m["per_case_auroc_macro"] is None
    assert m["per_case_auprc_macro"] is None
    assert m["n_cases_with_both_classes"] == 0


def test_per_case_macro_averages_across_cases():
    # caseA AUROC 1.0，caseC AUROC 0.0 → 宏平均 0.5
    rows = [
        _row("caseA", "a1", 0.1, False),
        _row("caseA", "a2", 0.9, True),
        _row("caseC", "c1", 0.9, False),
        _row("caseC", "c2", 0.1, True),
    ]
    m = compute_stratified_metrics(pd.DataFrame(rows))
    assert m["per_case_auroc_macro"] == 0.5
    assert m["n_cases_with_both_classes"] == 2
    assert "per_case_auprc_macro" in m
