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
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.contracts import validate_scores_df
from src.data.contract_dataloader import ContractDataset
from src.fusion.base import MODALITY_ORDER, FusionModule
from src.models.deep_svdd import DeepSVDD
from src.utils.seed import set_seed

LOG = logging.getLogger(__name__)


def _collate(batch: list[dict]) -> dict:
    """聚合 ContractDataset point 样本：modality tensor 堆叠，meta/label 保持 list。"""
    out: dict = {m: torch.stack([s[m] for s in batch]) for m in MODALITY_ORDER}
    out["sample_id"] = [s["meta"]["sample_id"] for s in batch]
    out["is_anomaly"] = [s["label"]["is_anomaly"] for s in batch]
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
) -> None:
    svdd.train()
    for epoch in range(epochs):
        total = 0.0
        n_batches = 0
        for batch in loader:
            x = fusion({m: batch[m] for m in MODALITY_ORDER})
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
) -> dict[str, float]:
    svdd.eval()
    scores: dict[str, float] = {}
    for batch in loader:
        x = fusion({m: batch[m] for m in MODALITY_ORDER})
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
    fusion = hydra.utils.instantiate(cfg.fusion, modality_dims=modality_dims)
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
    svdd.init_center(fusion({m: init_batch[m] for m in MODALITY_ORDER}))

    # instantiate 整份 optimizer config（lr + weight_decay 都来自 cfg.training.optimizer），
    # 不手搓、不只挑 lr——避免 weight_decay 等字段静默丢失。svdd.parameters() 作为
    # 运行时参数在 instantiate 时以 params= 补上（fusion 无可训练参数，只传 svdd）。
    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=svdd.parameters())

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=cfg.training.num_workers,
    )
    _train(fusion, svdd, train_loader, epochs=cfg.training.epochs, optimizer=optimizer)

    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=cfg.training.num_workers,
    )
    score_map = _infer(fusion, svdd, eval_loader)

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


if __name__ == "__main__":
    main()
