import json

from scripts.service_trace_features import read_trace_case


def test_read_trace_case_aggregates_spans_and_edges(tmp_path):
    payload = {
        "traces": [
            {
                "spans": [
                    {
                        "node_id": "a",
                        "trace_id": "t1",
                        "service_code": "gateway",
                        "start_timestamp_ms": 30_000,
                        "duration_ms": 10,
                        "is_error": False,
                        "children_node_ids": ["b"],
                    },
                    {
                        "node_id": "b",
                        "trace_id": "t1",
                        "service_code": "order",
                        "start_timestamp_ms": 30_001,
                        "duration_ms": 20,
                        "is_error": True,
                        "children_node_ids": [],
                    },
                ]
            }
        ]
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(payload))

    features, edges = read_trace_case(path)

    assert set(features.service) == {"gateway", "order"}
    assert features.loc[features.service.eq("order"), "error_rate"].iloc[0] == 1.0
    assert edges.to_dict("records") == [{"source_service": "gateway", "target_service": "order"}]
    assert edges.attrs["missing_child_count"] == 0


def test_read_trace_case_retains_missing_child_quality(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "spans": [
                            {
                                "node_id": "a",
                                "trace_id": "t1",
                                "service_code": "gateway",
                                "start_timestamp_ms": 0,
                                "duration_ms": 1,
                                "children_node_ids": ["missing"],
                            }
                        ]
                    }
                ]
            }
        )
    )

    _, edges = read_trace_case(path)

    assert edges.attrs["missing_child_count"] == 1
