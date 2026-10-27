import json
from pathlib import Path

import torch

import scripts.service_model as service_cli
import scripts.service_patchad_model as patchad_cli
from tests.test_service_model_smoke import TinyRuntime, synthetic_tables


class TinyPatchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(27, 3)
        self.second = torch.nn.Linear(27, 3)

    def forward(self, sequence):
        first = torch.softmax(self.first(sequence[:, :1]), dim=-1)
        second = torch.softmax(self.second(sequence[:, 1:]), dim=-1)
        return [first], [second], [first], [second], sequence


def test_patchad_cli_publishes_calibrated_results(tmp_path, monkeypatch):
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
    config = json.loads(Path("configs/service/tt_patchad_neural_v1.json").read_text())
    config.update(sequence_length=2, patch_sizes=[2], d_model=4, e_layers=1)
    config["training"].update(device="cpu", epochs=1, batch_size=1)
    config_path = tmp_path / "patchad.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(patchad_cli, "upstream_factory", lambda path, config: TinyPatchModel)
    output = tmp_path / "patchad"
    assert (
        patchad_cli.main(
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
    thresholds = json.loads((output / "run/thresholds.json").read_text())
    assert thresholds["primary_quantile"] == 0.99
