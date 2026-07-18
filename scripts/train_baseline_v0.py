#!/usr/bin/env python
"""Baseline v0 训练脚本：融合机制 + Deep SVDD 单类异常检测（Hydra 驱动）。

流程（One-Class 约定）：
1. 用 train.parquet（仅 Normal）训练，center 在正常表征上初始化
2. 推理 eval_all.parquet，输出每样本异常分数（距超球心距离²，higher=更异常）
3. 写出符合 scores_v0 契约的 parquet（含 case_id/anomaly_type 等诊断列）

eval_all 的特征 NaN 用 train.parquet 均值填补（fit_on_parquet），避免 eval 自身统计量泄漏。

融合机制与模型均通过 hydra.utils.instantiate 构造（cfg.fusion / cfg.model），
切换实现只需 CLI override（如 fusion=gated），不需要改这个脚本。

注意接口约束：fusion=xxx 可任意切换（都实现 forward/output_dim 统一接口）；
但 model=xxx 只对实现了 init_center/svdd_loss/score 三个方法的 One-Class 检测器成立——
本训练循环调这三个方法，换成非 SVDD 系模型（如 VAE）需另写训练循环，不能仅靠 config 切换。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.contracts import validate_scores_df
from src.contracts.endpoint_id_mapping import id_to_endpoint_key as _derive_id_to_endpoint_key
from src.data.contract_dataloader import ContractDataset
from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.base import MODALITY_ORDER, FusionModule
from src.models.deep_svdd import DeepSVDD
from src.utils.seed import set_seed

LOG = logging.getLogger(__name__)


def _collate(batch: list[dict]) -> dict:
    """聚合 ContractDataset point 样本：modality tensor 堆叠，meta/label 保持 list。

    endpoint_id 仅在 v1 数据（含该列）时存在于 meta；v0 数据不产出这个 key，
    batch 里也不会出现，此时 out 不含 "endpoint_id"，下游 fusion 调用走
    endpoint_id=None 的默认参数路径（向后兼容）。
    """
    out: dict = {m: torch.stack([s[m] for s in batch]) for m in MODALITY_ORDER}
    out["sample_id"] = [s["meta"]["sample_id"] for s in batch]
    out["is_anomaly"] = [s["label"]["is_anomaly"] for s in batch]
    if "endpoint_id" in batch[0]["meta"]:
        out["endpoint_id"] = torch.tensor([s["meta"]["endpoint_id"] for s in batch])
    return out


def _modality_dims(schema: dict) -> dict[str, int]:
    return {name: len(grp["columns"]) for name, grp in schema["feature_groups"].items()}


def _make_dataset(parquet_path: Path, schema_path: Path, fit_on: Path) -> ContractDataset:
    return ContractDataset(
        parquet_path=parquet_path,
        schema_path=schema_path,
        mode="point",
        nan_strategy="mean",
        fit_on_parquet=fit_on,
    )


def _train(
    fusion: FusionModule,
    svdd: DeepSVDD,
    loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    is_reliability_gate: bool,
) -> None:
    fusion.train()
    svdd.train()
    for epoch in range(epochs):
        total = 0.0
        n_batches = 0
        for batch in loader:
            # L0/L1/L2 的 forward() 不接受 endpoint_id 参数（未感知 per-endpoint
            # 路由概念），无条件传入会报 TypeError；只有 ReliabilityGatedFusion 接受。
            kwargs = {"endpoint_id": batch.get("endpoint_id")} if is_reliability_gate else {}
            x = fusion({m: batch[m] for m in MODALITY_ORDER}, **kwargs)
            loss = svdd.svdd_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n_batches += 1
        LOG.info("epoch %d/%d loss=%.6f", epoch + 1, epochs, total / max(n_batches, 1))


@torch.no_grad()
def _infer(
    fusion: FusionModule,
    svdd: DeepSVDD,
    loader: DataLoader,
    is_reliability_gate: bool,
) -> dict[str, float]:
    fusion.eval()
    svdd.eval()
    scores: dict[str, float] = {}
    for batch in loader:
        # 同 _train：L0/L1/L2 的 forward() 不接受 endpoint_id，无条件传会 TypeError。
        kwargs = {"endpoint_id": batch.get("endpoint_id")} if is_reliability_gate else {}
        x = fusion({m: batch[m] for m in MODALITY_ORDER}, **kwargs)
        s = svdd.score(x)
        for sid, val in zip(batch["sample_id"], s.tolist()):
            scores[sid] = val
    return scores


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    set_seed(cfg.seed)

    # hydra-core>=1.3 默认 hydra.job.chdir=False，不切换工作目录，
    # 因此 contract_dir/out 的相对路径解析规则与旧 argparse 版本完全一致
    # （相对于脚本被调用时的 cwd，不是 outputs/ 产物目录）。
    contract_dir = Path(cfg.contract_dir)
    train_pq = contract_dir / "train.parquet"
    eval_pq = contract_dir / "eval_all.parquet"
    schema_path = contract_dir / "schema.json"
    schema = json.loads(schema_path.read_text())

    train_ds = _make_dataset(train_pq, schema_path, fit_on=train_pq)
    eval_ds = _make_dataset(eval_pq, schema_path, fit_on=train_pq)

    modality_dims = _modality_dims(schema)
    # ReliabilityGatedFusion 需要额外的运行时 kwargs（endpoint_baseline_stats /
    # id_to_endpoint_key），这两个参数对 L0/L1/L2 是多余的——hydra.utils.instantiate
    # 遇到目标类 __init__ 不接受的多余关键字参数会报 TypeError，不能无条件传。
    # 按 _target_ 字符串分支是当前唯一需要特殊运行时参数的融合机制，YAGNI，
    # 若未来出现第二个需要特殊参数的机制再抽象成 factory。
    is_reliability_gate = (
        cfg.fusion._target_ == "src.fusion.reliability_gate.ReliabilityGatedFusion"
    )
    fusion_kwargs = {"modality_dims": modality_dims}
    if is_reliability_gate:
        baseline_stats = EndpointBaselineStats.load(contract_dir / "endpoint_baseline_stats.json")
        ep_to_svc = yaml.safe_load(
            (Path(__file__).parents[1] / "configs/contract/endpoint_to_service.yaml").read_text()
        )
        # 必须和 build_contract.py 里 endpoint_id_map 的派生方式一致（同一份
        # src/contracts/endpoint_id_mapping.py 源）——这里是那份映射的逆映射，
        # 顺序不一致会导致 endpoint_id 查到错的 endpoint_key，静默用错 baseline 统计量。
        id_to_endpoint_key = _derive_id_to_endpoint_key(ep_to_svc)
        fusion_kwargs["endpoint_baseline_stats"] = baseline_stats
        fusion_kwargs["id_to_endpoint_key"] = id_to_endpoint_key
    fusion = hydra.utils.instantiate(cfg.fusion, **fusion_kwargs)
    svdd = hydra.utils.instantiate(cfg.model, input_dim=fusion.output_dim)

    # 用全部 Normal 训练样本初始化超球心（One-Class：center 只见正常表征）
    LOG.info("初始化 SVDD 超球心，加载 %d 训练样本...", len(train_ds))
    if len(train_ds) > 10_000:
        LOG.warning(
            "训练集 %d 行，init_center 一次性加载全部数据到内存；如遇 OOM 请考虑增量初始化",
            len(train_ds),
        )
    init_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False, collate_fn=_collate)
    init_batch = next(iter(init_loader))
    init_kwargs = {"endpoint_id": init_batch.get("endpoint_id")} if is_reliability_gate else {}
    svdd.init_center(fusion({m: init_batch[m] for m in MODALITY_ORDER}, **init_kwargs))

    # instantiate 整份 optimizer config（lr + weight_decay 都来自 cfg.training.optimizer），
    # 不手搓、不只挑 lr——避免 weight_decay 等字段静默丢失。svdd.parameters() 与
    # fusion.parameters() 都要传入：L0（EarlyConcatFusion）确实无可训练参数，但
    # L1/L2 等融合机制引入了真实的 nn.Linear 层，遗漏会导致这些层永远停在随机初始化，
    # 训练全程不更新——只优化 SVDD 会让融合模块的消融实验结果失去意义。
    optimizer = hydra.utils.instantiate(
        cfg.training.optimizer, params=list(svdd.parameters()) + list(fusion.parameters())
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=cfg.training.num_workers,
    )
    _train(
        fusion,
        svdd,
        train_loader,
        epochs=cfg.training.epochs,
        optimizer=optimizer,
        is_reliability_gate=is_reliability_gate,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=cfg.training.num_workers,
    )
    score_map = _infer(fusion, svdd, eval_loader, is_reliability_gate=is_reliability_gate)

    eval_df = pd.read_parquet(eval_pq)
    out_df = pd.DataFrame(
        {
            "sample_id": eval_df["sample_id"].astype(str),
            "score": eval_df["sample_id"].map(score_map).astype(float),
            "y_true": eval_df["is_anomaly"].astype(int),
            "case_id": eval_df["case_id"],
            "endpoint_key": eval_df["endpoint_key"],
            "phase": eval_df["phase"],
            "anomaly_type": eval_df["anomaly_type"],
            "anomaly_level": eval_df["anomaly_level"],
            "is_endpoint_anomaly": eval_df["is_endpoint_anomaly"].astype(int),
            "label_granularity": eval_df["label_granularity"],
        }
    )

    validate_scores_df(out_df)

    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    LOG.info("写出 scores：%d 行 → %s", len(out_df), out_path)

    if cfg.fusion_checkpoint is not None:
        checkpoint_path = Path(cfg.fusion_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fusion.state_dict(), checkpoint_path)
        LOG.info("写出 fusion checkpoint → %s", checkpoint_path)


if __name__ == "__main__":
    main()
