import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
MINI_DATA_ROOT = REPO_ROOT / "tests/fixtures/mini_data_root"
REQUIRED_KEYS = {
    "contract_version",
    "window_size_s",
    "feature_dim",
    "feature_groups",
    "evaluation_strata",
}


def test_schema_json_keys_stable(tmp_path):
    out = tmp_path / "contract_v0"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v0.yaml"),
            "--data-root",
            str(MINI_DATA_ROOT),
            "--out-dir",
            str(out),
            "--seed",
            "1",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    schema = json.loads((out / "schema.json").read_text())
    assert REQUIRED_KEYS.issubset(schema.keys())
    assert schema["feature_dim"] == 18
    assert set(schema["feature_groups"].keys()) == {
        "endpoint_red",
        "service_metric",
        "service_log",
    }
