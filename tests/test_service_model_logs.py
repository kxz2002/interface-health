import copy
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.service.modeling.logs import fit_logs, prepare_logs


class FakeTokenizer:
    pad_token_id = 0
    name_or_path = "test-tokenizer"
    special_tokens_map = {"cls_token": "[CLS]", "sep_token": "[SEP]"}
    init_kwargs = {"revision": "test-revision"}

    def get_vocab(self):
        return {f"t{index}": 1000 + index for index in range(500)}

    def encode(self, text, add_special_tokens=False, truncation=False):
        assert not add_special_tokens and not truncation
        if text == "[DROP]":
            return []
        if text == "[SPECIAL]":
            return [111, 115]
        return [
            (
                1000 + int(word[1:])
                if word.startswith("t") and word[1:].isdigit()
                else 2000 + sum(map(ord, word))
            )
            for word in text.split()
        ]

    def num_special_tokens_to_add(self, pair=False):
        return 5

    def build_inputs_with_special_tokens(self, token_ids):
        return [111, 112, 113] + token_ids + [114, 115]

    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
        assert already_has_special_tokens
        return [int(token in {111, 112, 113, 114, 115}) for token in token_ids]


def row(sample_id, service, messages, quality="OBSERVED", event_count=None):
    items = [
        {"message": message, "count": 1} if isinstance(message, str) else message
        for message in messages
    ]
    return dict(
        sample_id=sample_id,
        canonical_service_id=service,
        messages_json=json.dumps(items),
        quality_status=quality,
        event_count=sum(item["count"] for item in items) if event_count is None else event_count,
    )


def tables(rows):
    frame = pd.DataFrame(rows)
    return SimpleNamespace(log=frame, grid=frame[["sample_id"]].copy())


def prepare(rows, fit_ids=()):
    data = tables(rows)
    state = fit_logs(data, set(fit_ids), {})
    return prepare_logs(data, state, FakeTokenizer(), {})


def test_frozen_service_templates_and_unknown_groups():
    rows = [
        row("a", "s1", ["request id 1 finished", "request id 2 finished"]),
        row("b", "s2", ["different payload"]),
        row("c", "s1", ["request id 9 finished"]),
        row("d", "s2", ["request id 9 finished"]),
        row("e", "s3", [{"message": "new one", "count": 3}, "new one", "new two"]),
    ]
    data = tables(rows)
    state = fit_logs(data, {"a", "b"}, {})
    before = copy.deepcopy(state.templates)
    snapshots = {
        service: [
            (cluster.cluster_id, cluster.get_template(), cluster.size)
            for cluster in miner.drain.clusters
        ]
        for service, miner in state.miners.items()
    }
    out = prepare_logs(data, state, FakeTokenizer(), {})
    assert out["group_ids"]["c"] == ["template:request id <*> finished"]
    assert out["group_ids"]["d"] == ["unmatched:request id 9 finished"]
    assert out["group_ids"]["e"] == ["unmatched:new one", "unmatched:new two"]
    assert out["stats"]["e"]["unmatched_count"] == 5
    assert state.templates == before
    assert snapshots == {
        service: [
            (cluster.cluster_id, cluster.get_template(), cluster.size)
            for cluster in miner.drain.clusters
        ]
        for service, miner in state.miners.items()
    }
    assert "s3" not in state.miners
    assert state.config["parser"] == "drain3"
    assert state.config["drain"]["masking_instructions"] == []
    assert state.config["drain"]["drain_sim_th"] == 0.4
    assert state.config["drain"]["drain_depth"] == 4


def test_fit_is_stable_ignores_counts_and_rejects_nonfit_ids():
    rows = [
        row("b", "s", ["request id 2 finished"]),
        row("a", "s", [{"message": "request id 1 finished", "count": 9000}]),
    ]
    state = fit_logs(tables(rows), {"a", "b"}, {})
    assert sum(cluster.size for cluster in state.miners["s"].drain.clusters) == 2
    assert state.state_hash == fit_logs(tables(list(reversed(rows))), {"a", "b"}, {}).state_hash
    with pytest.raises(ValueError, match="fit IDs"):
        fit_logs(tables(rows), {"unknown"}, {})
    data = tables(rows)
    data.splits = pd.DataFrame({"sample_id": ["a", "b"], "split": ["fit", "test"]})
    data.labels = pd.DataFrame({"sample_id": ["a", "b"], "label": [0, 0]})
    with pytest.raises(ValueError, match="known-normal fit"):
        fit_logs(data, {"b"}, {})


def test_missing_drain3_fails_explicitly(monkeypatch):
    original = importlib.import_module

    def missing(name, *args, **kwargs):
        if name.startswith("drain3"):
            raise ModuleNotFoundError("synthetic missing drain3")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="Drain3"):
        fit_logs(tables([row("a", "s", ["hello"])]), {"a"}, {})


def test_priority_overlap_and_round_robin_16():
    errors = [f"ERROR error {index:02d}" for index in range(6)]
    unknown = [f"unknown {index:02d}" for index in range(20)]
    data = tables([row("a", "s", errors + unknown)])
    state = fit_logs(data, set(), {})
    out = prepare_logs(data, state, FakeTokenizer(), {})
    assert out["texts"]["a"] == [f"error {index:02d}" for index in range(6)] + [
        f"unknown {index:02d}" for index in range(10)
    ]
    assert len(out["input_ids"]["a"]) == len(out["group_ids"]["a"]) == 16
    assert len(set(out["texts"]["a"])) == 16
    assert out["coverage"]["a"]["unselected_message_count"] == 10
    shuffled = tables([row("a", "s", list(reversed(errors + unknown)))])
    assert prepare_logs(shuffled, state, FakeTokenizer(), {}) == out


def test_round_robin_prioritizes_group_with_fewer_selected_messages():
    fit = row("a", "s", ["ERROR request id 1 finished", "ERROR request id 2 finished"])
    other = row(
        "b",
        "s",
        [f"ERROR request id {index} finished" for index in range(10)]
        + [f"new {index}" for index in range(5)],
    )
    out = prepare_logs(
        tables([fit, other]),
        fit_logs(tables([fit, other]), {"a"}, {}),
        FakeTokenizer(),
        {"max_messages": 10},
    )
    assert out["texts"]["b"][:4] == [f"request id {index} finished" for index in range(4)]
    assert out["texts"]["b"][4:9] == [f"new {index}" for index in range(5)]
    assert out["texts"]["b"][9] == "request id 4 finished"


def test_head_tail_128_and_empty_body():
    long = " ".join(f"t{index}" for index in range(300))
    out = prepare([row("a", "s", [long, "", "[DROP]", "INFO   "])])
    ids = out["input_ids"]["a"][0]
    assert len(ids) == 128
    assert ids == [111, 112, 113] + list(range(1000, 1082)) + list(range(1259, 1300)) + [114, 115]
    assert out["special_mask"]["a"][0] == [1, 1, 1] + [0] * 123 + [1, 1]
    assert sum(out["attention_mask"]["a"][0]) == 128
    assert out["stats"]["a"]["event_count"] == 1
    assert out["coverage"]["a"]["excluded_event_count"] == 3
    assert out["coverage"]["a"]["token_truncated_message_count"] == 1
    short = prepare([row("a", "s", ["t0 t1"])])
    assert short["input_ids"]["a"][0][:7] == [111, 112, 113, 1000, 1001, 114, 115]
    assert short["attention_mask"]["a"][0] == [1] * 7 + [0] * 121
    assert not any(
        attention and not special
        for attention, special in zip(
            short["attention_mask"]["a"][0][7:], short["special_mask"]["a"][0][7:]
        )
    )


def test_special_only_messages_excluded_before_selection_and_no_tokenizer_fallback():
    messages = ["[SPECIAL]"] + [f"body {index}" for index in range(16)]
    data = tables([row("a", "s", messages)])
    state = fit_logs(data, set(), {})
    out = prepare_logs(data, state, FakeTokenizer(), {})
    assert out["counts"]["a"] == 16
    assert "[SPECIAL]" not in out["texts"]["a"]
    assert out["stats"]["a"]["event_count"] == 16
    assert out["coverage"]["a"]["token_excluded_event_count"] == 1

    class BrokenTokenizer(FakeTokenizer):
        def encode(self, *args, **kwargs):
            raise ValueError("real tokenizer failure")

    with pytest.raises(ValueError, match="real tokenizer failure"):
        prepare_logs(data, state, BrokenTokenizer(), {})


def test_frozen_state_tampering_and_final_fit_rematch():
    data = tables([row("a", "s", ["request alpha completed", "request beta completed"])])
    state = fit_logs(data, {"a"}, {})
    out = prepare_logs(data, state, FakeTokenizer(), {})
    assert out["group_ids"]["a"] == ["template:request <*> completed"] * 2
    state.miners["s"].add_log_message("mutated template")
    with pytest.raises(ValueError, match="Frozen Drain3"):
        prepare_logs(data, state, FakeTokenizer(), {})


@pytest.mark.parametrize(
    "bad", ["{}", "not json", '[{"message":"x","count":true}]', '[{"message":"x","count":0}]']
)
def test_malformed_prepared_messages_fail_explicitly(bad):
    data = tables([row("a", "s", ["hello"])])
    data.log.loc[0, "messages_json"] = bad
    with pytest.raises(ValueError, match="Invalid log"):
        fit_logs(data, {"a"}, {})


def test_cleaning_preserves_uncertain_body_paths_numbers_and_stack():
    body = "failure /tmp/a42 503\n at pkg.Class(File.java:12)"
    out = prepare(
        [
            row(
                "a",
                "s",
                [
                    f"2026-10-21 12:00:00.123 ERROR 42 --- [main] pkg.Class : {body}",
                    "2026-99-99 12:00:00 ERROR keep this",
                    "business ERROR is payload",
                    {"message": "plain payload", "count": 1, "level": "ERROR"},
                ],
            )
        ]
    )
    assert out["texts"]["a"][0] == body
    assert "2026-99-99 12:00:00 ERROR keep this" in out["texts"]["a"]
    assert "business ERROR is payload" in out["texts"]["a"]
    assert out["coverage"]["a"]["error_message_count"] == 1
    long = "x" * 2200 + " /tmp/42"
    state = fit_logs(tables([row("a", "s", [long])]), {"a"}, {})
    assert long in state.templates["s"]
    assert state.config["fit_message_diagnostics"]["max_characters"] == len(long)


def test_full_window_stats_not_selected_stats():
    messages = [{"message": f"request id {index} finished", "count": 2} for index in range(20)] + [
        {"message": "unseen body", "count": 3}
    ]
    rows = [
        row("a", "s", ["request id 1 finished", "request id 2 finished"]),
        row("b", "s", messages, event_count=44),
    ]
    out = prepare(rows, {"a"})
    assert out["stats"]["b"] == dict(
        event_count=43, matched_template_count=1, max_matched_fraction=40 / 43, unmatched_count=3
    )
    assert out["counts"]["b"] == 16
    assert out["coverage"]["b"]["available_message_count"] == 21
    assert out["coverage"]["b"]["source_event_count"] == 44
    assert out["coverage"]["b"]["source_event_count_difference"] == 1
    assert out["coverage"]["b"]["selected_group_count"] == 2
    assert (
        out["coverage"]["b"]["selected_event_count"]
        + out["coverage"]["b"]["unselected_event_count"]
        == 43
    )


def test_quality_gates_fit_and_tokenization_and_distinguishes_empty():
    out = prepare(
        [
            row("a", "s", ["do not fit"], "COLLECTION_PARTIAL_INVALID_SOURCE_TIME"),
            row("b", "s", [], "NO_OBSERVATIONS_COLLECTION_PRESENT"),
            row("c", "s", [], "COLLECTION_MISSING"),
            row("d", "s", ["[DROP]", ""]),
            row("e", "s", ["new body"]),
        ],
        {"a", "b", "c", "d"},
    )
    assert out["group_ids"]["e"] == ["unmatched:new body"]
    assert out["coverage"]["a"]["reason"] == "source_quality_invalid"
    assert out["coverage"]["b"]["reason"] == "no_events_collection_present"
    assert out["coverage"]["c"]["reason"] == "collection_missing"
    assert out["coverage"]["d"]["reason"] == "messages_unusable"
    for sample_id in "abcd":
        assert out["masks"][sample_id] is False
        assert out["input_ids"][sample_id] == []
        assert out["stats"][sample_id]["max_matched_fraction"] is None
    assert out["stats"]["e"]["max_matched_fraction"] == 0
    data = tables([row("a", "s", ["hello"])])
    data.log = data.log.drop(columns="quality_status")
    with pytest.raises(ValueError, match="quality"):
        fit_logs(data, {"a"}, {})


def test_cache_binds_tokenizer_cleaning_config_fit_and_inputs():
    data = tables([row("a", "s", ["hello"])])
    fitted = fit_logs(data, {"a"}, {})
    out = prepare_logs(data, fitted, FakeTokenizer(), {})
    assert out["cache"]["cleaning_version"]
    assert out["cache"]["state_hash"] == fitted.state_hash
    assert out["cache"]["fit_ids"] == ["a"]
    assert out["cache"]["tokenizer"]["vocabulary_sha256"]
    changed = FakeTokenizer()
    changed.init_kwargs = {"revision": "different"}
    assert (
        prepare_logs(data, fitted, changed, {})["cache"]["cache_key"] != out["cache"]["cache_key"]
    )
    assert (
        prepare_logs(data, fitted, FakeTokenizer(), {"max_tokens": 64})["cache"]["cache_key"]
        != out["cache"]["cache_key"]
    )
    assert (
        prepare_logs(data, fit_logs(data, set(), {}), FakeTokenizer(), {})["cache"]["cache_key"]
        != out["cache"]["cache_key"]
    )
    changed_data = tables([row("a", "s", ["different"])])
    assert (
        prepare_logs(changed_data, fitted, FakeTokenizer(), {})["cache"]["cache_key"]
        != out["cache"]["cache_key"]
    )
    with pytest.raises(TypeError, match="tokenizer"):
        prepare_logs(data, fitted, object(), {})


def test_log_statistics_fit_only_service_means_and_scale_fallbacks():
    from src.service.modeling.logs import fit_log_statistics, transform_log_statistics

    rows = [
        row(sample_id, service, [{"message": "hello", "count": count}])
        for sample_id, service, count in [
            ("a", "A", 5),
            ("b", "A", 5),
            ("c", "B", 3),
            ("d", "B", 7),
            ("e", "A", 1000000),
            ("f", "C", 9),
        ]
    ]
    data = tables(rows)
    template_state = fit_logs(data, set("abcd"), {})
    logs = prepare_logs(data, template_state, FakeTokenizer(), {})
    state = fit_log_statistics(data, logs, set("abcd"), {})
    global_scale = np.std(np.log1p([5, 5, 3, 7]), ddof=0)
    params = state.fields["event_count"]["services"]["A"]
    assert params["mean"] == pytest.approx(np.log1p(5))
    assert params["scale"] == pytest.approx(global_scale)
    assert params["scale_source"] == "global"
    assert state.fields["max_matched_fraction"]["services"]["A"]["scale"] == 1
    assert not state.fields["event_count"]["services"]["C"]["available"]
    result = transform_log_statistics(data, logs, state)
    assert list(result["values"]) == [
        "sample_id",
        "event_count",
        "matched_template_count",
        "max_matched_fraction",
        "unmatched_count",
    ]
    values = result["values"].set_index("sample_id")
    masks = result["masks"].set_index("sample_id")
    assert values.at["e", "event_count"] == pytest.approx(
        (np.log1p(1000000) - np.log1p(5)) / global_scale
    )
    assert not masks.loc["f"].any()
    assert values.loc["f"].eq(0).all()
    data.log.loc[data.log.sample_id.eq("e"), "messages_json"] = (
        '[{"message":"hello","count":99999999}]'
    )
    changed = prepare_logs(data, template_state, FakeTokenizer(), {})
    assert fit_log_statistics(data, changed, set("abcd"), {}).state_hash == state.state_hash
    with pytest.raises(ValueError, match="fit IDs"):
        fit_log_statistics(data, logs, {"bad"}, {})


def test_log_statistics_no_fit_observation_and_source_binding():
    from src.service.modeling.logs import fit_log_statistics, transform_log_statistics

    data = tables(
        [row("a", "s", [], "NO_OBSERVATIONS_COLLECTION_PRESENT"), row("b", "s", ["hello"])]
    )
    template_state = fit_logs(data, {"a"}, {})
    logs = prepare_logs(data, template_state, FakeTokenizer(), {})
    state = fit_log_statistics(data, logs, {"a"}, {})
    result = transform_log_statistics(data, logs, state)
    assert not result["masks"].drop(columns="sample_id").any().any()
    assert result["values"].drop(columns="sample_id").eq(0).all().all()
    changed_tokenizer = FakeTokenizer()
    changed_tokenizer.init_kwargs = {"revision": "different"}
    changed = prepare_logs(data, template_state, changed_tokenizer, {})
    with pytest.raises(ValueError, match="binding"):
        transform_log_statistics(data, changed, state)
    data.splits = pd.DataFrame({"sample_id": ["a", "b"], "split": ["fit", "test"]})
    data.labels = pd.DataFrame({"sample_id": ["a", "b"], "label": [0, 0]})
    with pytest.raises(ValueError, match="known-normal fit"):
        fit_log_statistics(data, logs, {"b"}, {})


@pytest.mark.parametrize(
    "mutation", ["ids", "shape", "nan", "token_shape", "token_mask", "count", "list_ids"]
)
def test_assemble_prepared_validates_ids_and_shapes(mutation):
    from src.service.modeling.data import assemble_prepared

    data = tables([row("a", "s", ["hello"])])
    logs = prepare_logs(data, fit_logs(data, {"a"}, {}), FakeTokenizer(), {})
    numeric = {
        "values": pd.DataFrame({"sample_id": ["a"], "cpu_cores": [0.0]}),
        "masks": pd.DataFrame({"sample_id": ["a"], "cpu_cores": [True]}),
        "quality": pd.DataFrame(
            {"sample_id": ["a"], "resource_valid": [True], "execution_valid": [False]}
        ),
    }
    lists = SimpleNamespace(
        common_fit=("a",),
        common_calibration=(),
        common_main_test=(),
        extended_inference=(),
        extended_main_test=(),
        diagnostic_pre=(),
        diagnostic_positive=(),
        excluded=(),
    )
    assert assemble_prepared(data, numeric, logs, lists).sample_ids == ("a",)
    if mutation == "ids":
        logs["masks"]["wrong"] = logs["masks"].pop("a")
    elif mutation == "shape":
        numeric["masks"] = numeric["masks"].drop(columns="cpu_cores")
    elif mutation == "nan":
        numeric["values"].loc[0, "cpu_cores"] = np.nan
    elif mutation == "token_shape":
        logs["input_ids"]["a"][0].pop()
    elif mutation == "token_mask":
        logs["special_mask"]["a"][0] = [1] * 128
    elif mutation == "count":
        logs["counts"]["a"] += 1
    else:
        lists.common_fit = ("wrong",)
    with pytest.raises(ValueError):
        assemble_prepared(data, numeric, logs, lists)
