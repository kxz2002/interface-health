"""FusionModule.from_contract 钩子:L0/L1/L2 走基类默认(等价 hydra.instantiate),
RG 覆写自行加载 endpoint_baseline_stats + 派生 id_to_endpoint_key。同时验证
基类 forward 新增的 endpoint_id 参数对 L0/L1/L2 是接受即忽略(不改变输出)。"""

from pathlib import Path

import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

from src.contracts.endpoint_id_mapping import id_to_endpoint_key as _derive_id_to_key
from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.early_concat import EarlyConcatFusion
from src.fusion.gated import GatedFusion
from src.fusion.independent_concat import IndependentConcatFusion
from src.fusion.reliability_gate import ReliabilityGatedFusion

REPO_ROOT = Path(__file__).parents[1]
_DIMS = {"endpoint_red": 10, "service_metric": 5, "service_log": 3}


def _sample_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {m: torch.randn(batch_size, d) for m, d in _DIMS.items()}


def test_default_from_contract_matches_direct_instantiate(tmp_path):
    """L0 未覆写 from_contract,应走基类默认路径,构造结果与 hydra.instantiate 等价。
    contract_dir 在默认实现里被忽略(传一个不含任何 sidecar 的 tmp 目录也不报错)。"""
    cfg = OmegaConf.create({"_target_": "src.fusion.early_concat.EarlyConcatFusion"})
    fusion = EarlyConcatFusion.from_contract(cfg, contract_dir=tmp_path, modality_dims=_DIMS)
    assert isinstance(fusion, EarlyConcatFusion)
    assert fusion.output_dim == sum(_DIMS.values())


def test_l0_l1_l2_forward_accepts_and_ignores_endpoint_id():
    """基类 forward 统一新增 endpoint_id 参数后,L0/L1/L2 必须接受它但输出不受其影响
    (它们没有 per-endpoint 路由概念)。这样 train 脚本才能无条件传 endpoint_id。"""
    batch = _sample_batch()
    eid = torch.arange(batch["endpoint_red"].shape[0])
    for fusion in (
        EarlyConcatFusion(modality_dims=_DIMS),
        IndependentConcatFusion(modality_dims=_DIMS),
        GatedFusion(modality_dims=_DIMS),
    ):
        fusion.eval()
        with torch.no_grad():
            out_without = fusion(batch)
            out_with = fusion(batch, endpoint_id=eid)
        assert torch.equal(out_without, out_with)


def test_reliability_gate_from_contract_loads_baseline_stats(tmp_path):
    """RG 覆写 from_contract:从 contract_dir 读 endpoint_baseline_stats.json,
    并用 configs/contract/endpoint_to_service.yaml 派生 id_to_endpoint_key。
    验证加载的映射与权威派生函数一致、统计量覆盖到 fit 过的 endpoint。"""
    ep_to_svc = yaml.safe_load(
        (REPO_ROOT / "configs/contract/endpoint_to_service.yaml").read_text()
    )
    key = sorted(ep_to_svc)[0]
    df = pd.DataFrame(
        {
            "endpoint_key": [key, key],
            "endpoint_red__a": [0.1, 0.2],
            "endpoint_red__b": [0.3, 0.5],
            "service_metric__c": [0.4, 0.6],
        }
    )
    stats = EndpointBaselineStats(
        red_cols=["endpoint_red__a", "endpoint_red__b"], svc_cols=["service_metric__c"]
    )
    stats.fit(df)
    stats.save(tmp_path / "endpoint_baseline_stats.json")

    cfg = OmegaConf.create(
        {
            "_target_": "src.fusion.reliability_gate.ReliabilityGatedFusion",
            "branch_dim": 16,
            "dropout": 0.1,
        }
    )
    fusion = ReliabilityGatedFusion.from_contract(cfg, contract_dir=tmp_path, modality_dims=_DIMS)
    assert isinstance(fusion, ReliabilityGatedFusion)
    assert fusion.output_dim == 16
    assert fusion._id_to_key == _derive_id_to_key(ep_to_svc)
    assert key in fusion._baseline.fitted_endpoints()
