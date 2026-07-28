"""Deviation-Weighted Fusion（逐特征偏离量加权融合）。

与 `ReliabilityGatedFusion` 的核心区别：本类零可学习参数（无 `nn.Linear`
encoder、无 gate_mlp、无 softmax/sigmoid 分支级门控），权重直接由
`EndpointBaselineStats.branch_stats()` 给出的 mean/std 算出，粒度是**逐特征**
（每一维单独一个 sigmoid(|z|-threshold) 权重）而非 RG 的**分支级**（整个
ep/svc 分支共享一个标量权重）。forward 输出直接是加权后的原始特征拼接
（维度 = ep_dim + svc_dim），不经过任何 encoder/value 投影——设计目标是
验证"逐特征加权"这一最小改动本身能否缓解信号被平均稀释的问题（entry 017），
不引入模型容量变化作为混淆变量，故不参考 RG 的复杂度。
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
import yaml
from omegaconf import DictConfig

from src.contracts.endpoint_id_mapping import id_to_endpoint_key as _derive_id_to_endpoint_key
from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.base import MODALITY_ORDER, FusionModule

# |z|>此阈值算作"该特征显著偏离"，复用 RG 的既有显著性标准（见
# reliability_gate.py::_DEVIATION_THRESHOLD），非分支级门控输入而是
# 直接作用于逐特征权重公式 sigmoid(|z|-threshold)。
_DEFAULT_THRESHOLD = 2.0

# endpoint_to_service.yaml 是 id_to_endpoint_key 反查表的唯一权威来源，必须与
# build_contract.py 派生 endpoint_id 列时用的同一份 sorted() 逻辑一致
# (src/contracts/endpoint_id_mapping.py)，与 reliability_gate.py 同款路径计算。
_EP_TO_SVC_PATH = Path(__file__).resolve().parents[2] / "configs/contract/endpoint_to_service.yaml"


class DeviationWeightedFusion(FusionModule):
    def __init__(
        self,
        modality_dims: dict[str, int],
        endpoint_baseline_stats: EndpointBaselineStats,
        id_to_endpoint_key: dict[int, str],
        red_cols: list[str],
        svc_cols: list[str],
        threshold: float = _DEFAULT_THRESHOLD,
    ):
        super().__init__()
        missing = set(MODALITY_ORDER) - set(modality_dims)
        extra = set(modality_dims) - set(MODALITY_ORDER)
        if missing or extra:
            raise ValueError(
                f"modality_dims keys {set(modality_dims)} must match MODALITY_ORDER {MODALITY_ORDER}"
            )
        self._baseline = endpoint_baseline_stats
        self._id_to_key = dict(id_to_endpoint_key)
        self._threshold = threshold
        self._ep_dim = modality_dims["endpoint_red"]
        self._svc_dim = modality_dims["service_metric"] + modality_dims["service_log"]

        # branch_stats() 返回的 mean/std 向量按 stats 对象内部 fit 顺序排列，
        # 而退化列 mask 是按 red_cols/svc_cols 的位置顺序构建——两者长度不一致时
        # （调用方传错列表，或未来 schema.json 列序与 fit 顺序漂移）会静默错位，
        # 只在 forward 里 torch.where 广播失败时才报出无意义的 shape 错误。
        # 在构造时就 fail fast，同项目历史上 Normalizer 零方差错位 bug 同类教训。
        if len(red_cols) != self._ep_dim:
            raise ValueError(
                f"len(red_cols)={len(red_cols)} must equal modality_dims['endpoint_red']={self._ep_dim}"
            )
        if len(svc_cols) != self._svc_dim:
            raise ValueError(
                f"len(svc_cols)={len(svc_cols)} must equal "
                f"modality_dims['service_metric']+modality_dims['service_log']={self._svc_dim}"
            )
        # 长度相等只排除了数量错配，排不掉顺序错配（schema.json 列序与
        # EndpointBaselineStats 内部 fit 顺序独立派生，理论上可能同长度但顺序不同，
        # 此时 branch_stats() 的 mean/std 会被错位应用到 red_cols/svc_cols 的
        # 错误位置，计算不报错但结果是错的）——按值比对两者顺序，堵住这个此前
        # 仅由注释警告、未被代码强制的缺口。
        if list(red_cols) != endpoint_baseline_stats.red_cols:
            raise ValueError(
                "red_cols 与 endpoint_baseline_stats 内部 fit 顺序不一致，"
                f"传入={list(red_cols)!r}，fit 顺序={endpoint_baseline_stats.red_cols!r}——"
                "检查 schema.json 的 feature_groups 列序是否与 build_contract.py "
                "fit EndpointBaselineStats 时的列序一致"
            )
        if list(svc_cols) != endpoint_baseline_stats.svc_cols:
            raise ValueError(
                "svc_cols 与 endpoint_baseline_stats 内部 fit 顺序不一致，"
                f"传入={list(svc_cols)!r}，fit 顺序={endpoint_baseline_stats.svc_cols!r}——"
                "检查 schema.json 的 feature_groups 列序是否与 build_contract.py "
                "fit EndpointBaselineStats 时的列序一致"
            )

        # 退化列（EndpointBaselineStats 全局 fallback 到 1e-9 std 的列）的权重必须
        # 固定为1，不能走 sigmoid(|z|-threshold) 公式——否则一个从未真正偏离过的
        # 常数列，只因分母被兜底成 1e-9 而在 z 计算里引入数值噪声，被误判为需要
        # 衰减（见 test_degenerate_column_weight_fixed_to_one 的手算示例）。
        # register_buffer 而非普通属性：需要跟随 .to(device) 迁移，但不是可学习参数。
        ep_degenerate = set(endpoint_baseline_stats.degenerate_columns("ep"))
        svc_degenerate = set(endpoint_baseline_stats.degenerate_columns("svc"))
        self.register_buffer(
            "_ep_degenerate_mask",
            torch.tensor([c in ep_degenerate for c in red_cols], dtype=torch.bool),
        )
        self.register_buffer(
            "_svc_degenerate_mask",
            torch.tensor([c in svc_degenerate for c in svc_cols], dtype=torch.bool),
        )

    def _weighted(
        self,
        raw: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        degenerate_mask: torch.Tensor,
    ) -> torch.Tensor:
        z = (raw - mean) / std
        w = torch.sigmoid(z.abs() - self._threshold)
        w = torch.where(degenerate_mask, torch.ones_like(w), w)
        return raw * w

    def forward(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        raw_ep = modality_dict["endpoint_red"]
        raw_svc = torch.cat([modality_dict["service_metric"], modality_dict["service_log"]], dim=-1)

        # 没有 endpoint 信息时退化为纯 concat（等价 L0），兼容不携带 endpoint_id
        # 的既有调用路径（spec §2.5）——与 RG 的同款早退语义一致。
        if endpoint_id is None:
            return torch.cat([raw_ep, raw_svc], dim=-1)

        device = raw_ep.device
        weighted_eps, weighted_svcs = [], []
        for i in range(raw_ep.shape[0]):
            eid = int(endpoint_id[i].item())
            if eid not in self._id_to_key:
                raise KeyError(
                    f"endpoint_id={eid} 不在 id_to_endpoint_key 映射中，"
                    "检查上游 endpoint_id 编码是否与 fusion 构造时传入的映射一致"
                )
            key = self._id_to_key[eid]
            ep_mean, ep_std = self._baseline.branch_stats(key, "ep")
            svc_mean, svc_std = self._baseline.branch_stats(key, "svc")
            weighted_eps.append(
                self._weighted(
                    raw_ep[i],
                    torch.as_tensor(ep_mean, dtype=torch.float32, device=device),
                    torch.as_tensor(ep_std, dtype=torch.float32, device=device),
                    self._ep_degenerate_mask,
                )
            )
            weighted_svcs.append(
                self._weighted(
                    raw_svc[i],
                    torch.as_tensor(svc_mean, dtype=torch.float32, device=device),
                    torch.as_tensor(svc_std, dtype=torch.float32, device=device),
                    self._svc_degenerate_mask,
                )
            )
        return torch.cat([torch.stack(weighted_eps), torch.stack(weighted_svcs)], dim=-1)

    @property
    def output_dim(self) -> int:
        return self._ep_dim + self._svc_dim

    @classmethod
    def from_contract(
        cls, cfg: DictConfig, *, contract_dir: Path, modality_dims: dict[str, int]
    ) -> "DeviationWeightedFusion":
        """从 contract_dir 加载 per-endpoint 基线统计量 + schema 列名，派生
        id_to_endpoint_key 反查表，交给 hydra.utils.instantiate 注入这些运行时对象。
        与 ReliabilityGatedFusion.from_contract 同一模式（entry 015 收束的
        构造解耦），训练脚本对"本类需要什么"一无所知即可正确构造。"""
        sidecar = Path(contract_dir) / "endpoint_baseline_stats.json"
        if not sidecar.exists():
            raise FileNotFoundError(
                f"{sidecar} 不存在——DeviationWeightedFusion 需要 contract 构建时开启"
                " fit_endpoint_baseline_stats=true（见 configs/contract/v1_expanded_pool.yaml），"
                f"检查 {contract_dir} 是否是用 fit_endpoint_baseline_stats=false 的配置"
                "（如 v1.yaml）构建的"
            )
        baseline_stats = EndpointBaselineStats.load(sidecar)

        schema_path = Path(contract_dir) / "schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(
                f"{schema_path} 不存在——DeviationWeightedFusion.from_contract 需要 "
                f"contract_dir 是一份完整的 build_contract.py 产物，检查 {contract_dir} "
                "是否指向了正确的 contract 输出目录"
            )
        schema = json.loads(schema_path.read_text())
        try:
            feature_groups = schema["feature_groups"]
            red_cols = feature_groups["endpoint_red"]["columns"]
            svc_cols = (
                feature_groups["service_metric"]["columns"]
                + feature_groups["service_log"]["columns"]
            )
        except KeyError as e:
            raise KeyError(
                f"{schema_path} 缺少必需字段 {e}——schema.json 版本可能与当前"
                " DeviationWeightedFusion 期望的 feature_groups 结构（endpoint_red/"
                "service_metric/service_log）不兼容，检查 contract 构建时用的配置版本"
            ) from e

        if not _EP_TO_SVC_PATH.exists():
            raise FileNotFoundError(
                f"{_EP_TO_SVC_PATH} 不存在——DeviationWeightedFusion 需要该文件派生 "
                "id_to_endpoint_key 反查表，检查是否在 repo 根目录下运行"
            )
        ep_to_svc = yaml.safe_load(_EP_TO_SVC_PATH.read_text())
        id_to_key = _derive_id_to_endpoint_key(ep_to_svc)
        return hydra.utils.instantiate(
            cfg,
            modality_dims=modality_dims,
            endpoint_baseline_stats=baseline_stats,
            id_to_endpoint_key=id_to_key,
            red_cols=red_cols,
            svc_cols=svc_cols,
        )
