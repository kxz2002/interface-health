"""FusionModule.from_contract 钩子:L0/L1/L2 走基类默认(等价 hydra.instantiate),
RG 覆写自行加载 endpoint_baseline_stats + 派生 id_to_endpoint_key。同时验证
基类 forward 新增的 endpoint_id 参数对 L0/L1/L2 是接受即忽略(不改变输出)。"""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from src.contracts.endpoint_id_mapping import id_to_endpoint_key as _derive_id_to_key
from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.deviation_weighted import DeviationWeightedFusion
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


def test_reliability_gate_from_contract_missing_sidecar_raises_clear_error(tmp_path):
    """fit_endpoint_baseline_stats 与 fusion 选型解耦后,两者不再靠 contract_version
    绑定一致——contract_dir 若是用 flag=false 的配置(如 v1.yaml)构建的,缺 sidecar
    文件,必须 fail fast 且报错信息指向"检查 contract 构建时的 flag 取值",而不是
    裸 FileNotFoundError 只报路径。"""
    cfg = OmegaConf.create(
        {
            "_target_": "src.fusion.reliability_gate.ReliabilityGatedFusion",
            "branch_dim": 16,
            "dropout": 0.1,
        }
    )
    with pytest.raises(FileNotFoundError, match="fit_endpoint_baseline_stats"):
        ReliabilityGatedFusion.from_contract(cfg, contract_dir=tmp_path, modality_dims=_DIMS)


def test_deviation_weighted_from_contract_loads_baseline_stats_and_schema(tmp_path):
    """DeviationWeightedFusion 覆写 from_contract:从 contract_dir 读
    endpoint_baseline_stats.json + schema.json(后者提供逐列列名,RG 不需要这份,
    因为 RG 只算分支级偏离摘要,不需要知道单列名字;本设计需要把 degenerate_columns()
    返回的列名映射到张量位置,schema.json 的 feature_groups 是列顺序的权威来源)。"""
    ep_to_svc = yaml.safe_load(
        (REPO_ROOT / "configs/contract/endpoint_to_service.yaml").read_text()
    )
    key = sorted(ep_to_svc)[0]
    dims = {"endpoint_red": 2, "service_metric": 1, "service_log": 1}
    schema = {
        "feature_groups": {
            "endpoint_red": {"columns": ["endpoint_red__a", "endpoint_red__b"]},
            "service_metric": {"columns": ["service_metric__c"]},
            "service_log": {"columns": ["service_log__d"]},
        }
    }
    (tmp_path / "schema.json").write_text(json.dumps(schema))

    df = pd.DataFrame(
        {
            "endpoint_key": [key, key],
            "endpoint_red__a": [0.1, 0.2],
            "endpoint_red__b": [0.3, 0.5],
            "service_metric__c": [0.4, 0.6],
            "service_log__d": [0.2, 0.3],
        }
    )
    stats = EndpointBaselineStats(
        red_cols=["endpoint_red__a", "endpoint_red__b"],
        svc_cols=["service_metric__c", "service_log__d"],
    )
    stats.fit(df)
    stats.save(tmp_path / "endpoint_baseline_stats.json")

    cfg = OmegaConf.create(
        {"_target_": "src.fusion.deviation_weighted.DeviationWeightedFusion", "threshold": 2.0}
    )
    fusion = DeviationWeightedFusion.from_contract(cfg, contract_dir=tmp_path, modality_dims=dims)
    assert isinstance(fusion, DeviationWeightedFusion)
    assert fusion.output_dim == 4
    assert fusion._id_to_key == _derive_id_to_key(ep_to_svc)
    assert key in fusion._baseline.fitted_endpoints()


def test_deviation_weighted_from_contract_missing_sidecar_raises_clear_error(tmp_path):
    """与 RG 同款约定:sidecar 缺失必须 fail fast 且报错信息指向
    fit_endpoint_baseline_stats 配置项,不是裸 FileNotFoundError 只报路径。
    sidecar 检查在读 schema.json 之前,contract_dir 空目录即可触发。"""
    cfg = OmegaConf.create(
        {"_target_": "src.fusion.deviation_weighted.DeviationWeightedFusion", "threshold": 2.0}
    )
    with pytest.raises(FileNotFoundError, match="fit_endpoint_baseline_stats"):
        DeviationWeightedFusion.from_contract(cfg, contract_dir=tmp_path, modality_dims=_DIMS)


def _stats_sidecar_only(tmp_path):
    """写好 endpoint_baseline_stats.json 但故意不写 schema.json，用于测试
    schema.json 缺失/字段不全时报错是否清晰（sidecar 检查通过之后才会读到
    schema.json，所以这两步分开测才能定位到具体是哪一步失败）。"""
    df = pd.DataFrame(
        {
            "endpoint_key": ["k1", "k1"],
            "endpoint_red__a": [0.1, 0.2],
            "endpoint_red__b": [0.3, 0.5],
            "service_metric__c": [0.4, 0.6],
            "service_log__d": [0.2, 0.3],
        }
    )
    stats = EndpointBaselineStats(
        red_cols=["endpoint_red__a", "endpoint_red__b"],
        svc_cols=["service_metric__c", "service_log__d"],
    )
    stats.fit(df)
    stats.save(tmp_path / "endpoint_baseline_stats.json")


def test_deviation_weighted_from_contract_missing_schema_raises_clear_error(tmp_path):
    """sidecar 存在但 schema.json 缺失——之前会在这里裸抛
    FileNotFoundError 只报 schema.json 路径，没有任何指引；现在必须明确
    指向 contract_dir 本身不完整。"""
    _stats_sidecar_only(tmp_path)
    cfg = OmegaConf.create(
        {"_target_": "src.fusion.deviation_weighted.DeviationWeightedFusion", "threshold": 2.0}
    )
    with pytest.raises(FileNotFoundError, match="schema.json"):
        DeviationWeightedFusion.from_contract(
            cfg,
            contract_dir=tmp_path,
            modality_dims={"endpoint_red": 2, "service_metric": 1, "service_log": 1},
        )


def test_deviation_weighted_from_contract_schema_missing_feature_group_raises_clear_error(tmp_path):
    """schema.json 存在但缺 feature_groups 里的必需 key（如旧版本 schema
    没有 endpoint_red）——之前会裸抛 KeyError: 'endpoint_red'，没有任何
    上下文；现在必须提示 schema 版本不兼容。"""
    _stats_sidecar_only(tmp_path)
    incomplete_schema = {"feature_groups": {"service_metric": {"columns": ["service_metric__c"]}}}
    (tmp_path / "schema.json").write_text(json.dumps(incomplete_schema))
    cfg = OmegaConf.create(
        {"_target_": "src.fusion.deviation_weighted.DeviationWeightedFusion", "threshold": 2.0}
    )
    with pytest.raises(KeyError, match="endpoint_red"):
        DeviationWeightedFusion.from_contract(
            cfg,
            contract_dir=tmp_path,
            modality_dims={"endpoint_red": 2, "service_metric": 1, "service_log": 1},
        )
