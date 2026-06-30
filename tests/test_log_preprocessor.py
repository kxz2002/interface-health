import logging
import tempfile
from pathlib import Path

from src.preprocessors.log_preprocessor import LogPreprocessor

REPO_ROOT = Path(__file__).parents[1]
MINI_LOGS = REPO_ROOT / "tests/fixtures/mini_logs"
SAMPLE_LOG = MINI_LOGS / "ts-route-service/sample.log"


def test_log_output_columns():
    pre = LogPreprocessor()
    assert set(pre.get_feature_columns()) == {
        "service_log__event_rate",
        "service_log__error_ratio",
        "service_log__template_diversity",
    }


def test_log_fit_only_uses_normal_logs(tmp_path):
    """Drain3 模板必须从 Normal case 训练，fit 后产生 drain3 state 文件。"""
    pre = LogPreprocessor(drain3_state_path=tmp_path / "drain.bin")
    pre.fit([SAMPLE_LOG])
    assert (tmp_path / "drain.bin").exists()


def test_log_transform_outputs_service_window_rows(tmp_path):
    pre = LogPreprocessor(drain3_state_path=tmp_path / "drain.bin")
    pre.fit([SAMPLE_LOG])
    df = pre.transform(MINI_LOGS, case_meta={"case_id": "Normal_planA"})
    assert "service_name" in df.columns
    assert "timestamp_window_ms" in df.columns
    assert (df["service_name"] == "ts-route-service").all()


def test_log_error_ratio_when_no_events_is_zero(tmp_path):
    """无事件窗口的 error_ratio 必须是 0 而非 NaN。"""
    pre = LogPreprocessor(drain3_state_path=tmp_path / "drain.bin")
    pre.fit([SAMPLE_LOG])
    df = pre.transform(MINI_LOGS, case_meta={"case_id": "Normal_planA"})
    empty_rows = df[df["service_log__event_rate"] == 0]
    assert len(empty_rows) > 0
    assert (empty_rows["service_log__error_ratio"] == 0).all()
    assert (empty_rows["service_log__template_diversity"] == 0).all()


def test_canonical_service_name_strips_pod_suffix():
    """pod 目录名后缀应被剥离得到 canonical service name。"""
    assert (
        LogPreprocessor._canonical_service_name("ts-route-service-7a4b5c6d8e-ab9zx")
        == "ts-route-service"
    )
    assert (
        LogPreprocessor._canonical_service_name("ts-travel-service-6f467bc998-jt5dd")
        == "ts-travel-service"
    )


def test_canonical_service_name_no_suffix_unchanged():
    """不含 pod 后缀的目录名保持不变。"""
    assert LogPreprocessor._canonical_service_name("ts-route-service") == "ts-route-service"


def test_transform_warns_when_not_fitted(caplog):
    """未 fit 时 transform 应发出 warning。"""
    pre = LogPreprocessor()  # no fit
    with caplog.at_level(logging.WARNING, logger="src.preprocessors.log_preprocessor"):
        with tempfile.TemporaryDirectory() as tmpdir:
            pre.transform(Path(tmpdir), case_meta={"case_id": "test"})
    assert any("未 fit" in r.message or "template_id" in r.message for r in caplog.records)


def test_find_service_dirs_nested_layout(tmp_path):
    """nested: log_root/<run-id>/<svc-pod>/*.log"""
    run_dir = tmp_path / "run_20260609"
    svc_dir = run_dir / "ts-order-service-abc123-xyz12"
    svc_dir.mkdir(parents=True)
    (svc_dir / "ts-order-service.log").write_text("dummy")

    result = LogPreprocessor._find_service_dirs(tmp_path)
    assert result == [svc_dir]


def test_parse_line_timezone_converts_to_utc():
    """CST 10:41:43 → UTC 02:41:43（差 8h），验证 epoch ms 对应 UTC 时间。"""
    import pandas as pd

    log_pre = LogPreprocessor(log_timezone="Asia/Shanghai")
    line = "2026-06-09 10:41:43.491  INFO 1 --- [thread] logger : message"
    result = log_pre._parse_line(line)
    assert result is not None
    ts_ms, level, _ = result
    utc_time = pd.to_datetime(ts_ms, unit="ms", utc=True)
    assert utc_time.hour == 2
    assert utc_time.minute == 41
    assert level == "INFO"
