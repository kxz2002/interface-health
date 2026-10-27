import json
from pathlib import Path

import pytest
import torch

import scripts.service_model as service_cli
import scripts.service_reconstruction_model as reconstruction_cli
from tests.test_service_model_smoke import TinyRuntime, synthetic_tables
from tests.test_service_reconstruction_baseline import TinyReconstruction


@pytest.mark.parametrize("name", ["kanad", "timesnet"])
def test_reconstruction_cli_publishes_calibrated_results(tmp_path, monkeypatch, name):
    torch.set_num_threads(1)
    prep_config = tmp_path / "prepare.json"
    prep_config.write_text(json.dumps({"training": {"epochs": 1, "device": "cpu"}}))
    monkeypatch.setattr(
        service_cli, "load_inputs", lambda preparation, labels, cfg: synthetic_tables(cfg)
    )
    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    prepared = tmp_path / "prepared"
    assert (
        service_cli.main(
            [
                "prepare",
                "--config",
                str(prep_config),
                "--tracking-uri",
                tracking,
                "--output",
                str(prepared),
                "--preparation",
                "synthetic",
                "--label-bundle",
                "synthetic",
            ],
            runtime_factory=TinyRuntime(),
        )
        == 0
    )
    config = json.loads(Path(f"configs/service/tt_{name}_reconstruction_v1.json").read_text())
    config["sequence_length"] = 2
    config["training"].update(device="cpu", epochs=1, batch_size=2)
    if name == "timesnet":
        config["architecture"]["top_k"] = 1
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(
        reconstruction_cli, "upstream_factory", lambda path, cfg: TinyReconstruction
    )
    batches = []
    actual_score = reconstruction_cli.score_reconstruction

    def record_batch(model, inputs, ids, batch_size):
        batches.append(batch_size)
        return actual_score(model, inputs, ids, batch_size)

    monkeypatch.setattr(reconstruction_cli, "score_reconstruction", record_batch)
    output = tmp_path / "output"
    assert (
        reconstruction_cli.main(
            [
                "--config",
                str(config_path),
                "--prepared",
                str(prepared),
                "--upstream",
                "synthetic",
                "--output",
                str(output),
                "--tracking-uri",
                tracking,
            ]
        )
        == 0
    )
    assert service_cli.check_run(output)["status"] == "success"
    report = json.loads((output / "evaluation/report.json").read_text())
    assert report["coverage"]["missing_ids"] == []
    assert batches == ([1, 1] if name == "timesnet" else [2, 2])
