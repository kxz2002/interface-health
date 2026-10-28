"""Small, protocol-safe service-graph baseline adapters.

These implementations reproduce the input granularity of published
microservice methods (service graph plus multimodal windows) without copying
unlicensed upstream code. They are explicitly labelled adaptations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from scripts.service_model import read_prepared
from scripts.service_revaluate_temporal import accepted_bundle, rebind
from scripts.service_trace_features import read_trace_case

SEED = 42
HISTORY = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arrays(prepared, tables, raw_root: Path, fit_case_ids: set[str]):
    identity = prepared.identity.set_index("sample_id")
    value = prepared.numeric["values"].set_index("sample_id").fillna(0.0)
    mask = prepared.numeric["masks"].set_index("sample_id").fillna(0.0)
    numeric = value.join(mask, lsuffix="_value", rsuffix="_mask")
    trace_cols = ["trace_count", "span_count", "duration_mean_ms", "duration_p95_ms", "error_rate"]
    services = sorted(identity.canonical_service_id.unique())
    service_index = {name: i for i, name in enumerate(services)}
    case_arrays = {}
    for case in sorted(identity.case_id.unique()):
        paths = list((raw_root / case / "trace_data").glob("*skywalking*.json"))
        if not paths or case in set(tables.protocol["evaluation_excluded_cases"]):
            continue
        payload = json.loads(paths[0].read_text())
        origin = int(payload["metadata"]["expected_windows"][0][0])
        trace, edges = read_trace_case(paths[0], bucket_origin_ms=origin)
        trace_map = {(int(row.bucket), row.service): row for row in trace.itertuples()}
        rows = identity[identity.case_id.eq(case)].sort_values(
            ["window_start_ms", "canonical_service_id"]
        )
        windows = sorted(rows.window_start_ms.unique())
        x = np.zeros(
            (len(windows), len(services), len(numeric.columns) + len(trace_cols)), dtype=np.float32
        )
        ids = []
        for wi, start in enumerate(windows):
            index = int((start - origin) // 15_000)
            for sample_id, row in rows[rows.window_start_ms.eq(start)].iterrows():
                si = service_index[row.canonical_service_id]
                ids.append((wi, si, sample_id))
                x[wi, si, : len(numeric.columns)] = numeric.loc[sample_id].to_numpy(
                    dtype=np.float32
                )
                feature = trace_map.get((index, row.canonical_service_id))
                if feature is not None:
                    x[wi, si, len(numeric.columns) :] = np.asarray(
                        [getattr(feature, col) for col in trace_cols], dtype=np.float32
                    )
        adjacency = np.eye(len(services), dtype=np.float32)
        for edge in edges.itertuples(index=False):
            if edge.source_service in service_index and edge.target_service in service_index:
                adjacency[
                    service_index[edge.source_service], service_index[edge.target_service]
                ] = 1.0
        case_arrays[case] = dict(x=x, ids=ids, windows=windows, adjacency=adjacency)
    fit_adjacency = np.eye(len(services), dtype=np.float32)
    for case in fit_case_ids:
        if case in case_arrays:
            fit_adjacency = np.maximum(fit_adjacency, case_arrays[case]["adjacency"])
    fit_adjacency /= fit_adjacency.sum(axis=1, keepdims=True).clip(min=1.0)
    fit_cases = [case for case in fit_case_ids if case in case_arrays]
    if fit_cases:
        offset = len(numeric.columns)
        trace_fit = np.concatenate(
            [case_arrays[case]["x"][:, :, offset:] for case in fit_cases], axis=0
        )
        trace_mean = trace_fit.mean(axis=(0, 1), keepdims=True)
        trace_std = trace_fit.std(axis=(0, 1), keepdims=True).clip(min=1e-6)
        for case_data in case_arrays.values():
            case_data["x"][:, :, offset:] = (case_data["x"][:, :, offset:] - trace_mean) / trace_std
            case_data["adjacency"] = fit_adjacency
    return case_arrays, service_index


class GraphReconstruction(nn.Module):
    def __init__(self, dim: int, hidden: int, adjacency: torch.Tensor, mode: str):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.mode = mode
        self.encoder = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.temporal = nn.GRU(hidden, hidden, batch_first=True) if mode == "anofusion" else None
        self.decoder = nn.Linear(hidden, dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: batch × history × node × feature
        current = sequence[:, -1]
        encoded = self.encoder(current)
        encoded = torch.einsum("ij,bjh->bih", self.adjacency, encoded)
        if self.temporal is not None:
            history = self.encoder(sequence).permute(0, 2, 1, 3)
            encoded = self.temporal(history.reshape(-1, history.shape[2], history.shape[3]))[0][
                :, -1
            ]
            encoded = encoded.reshape(sequence.shape[0], sequence.shape[2], -1)
        return self.decoder(encoded)


class GraphSVDD(nn.Module):
    def __init__(self, dim: int, hidden: int, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.encoder = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        z = self.encoder(current)
        return torch.einsum("ij,bjh->bih", self.adjacency, z)


def _windows(case_data, fit_windows: set[int], history: int):
    x = case_data["x"]
    for wi in range(history - 1, len(x)):
        if wi in fit_windows:
            yield wi, x[wi - history + 1 : wi + 1]


def run_model(
    name,
    arrays,
    fit_cases,
    cal_cases,
    test_cases,
    labels,
    allowed_calibration_ids,
    allowed_test_ids,
    seed,
):
    _set_seed(seed)
    sample = next(iter(arrays.values()))
    dim = sample["x"].shape[-1]
    nodes = sample["x"].shape[1]
    adjacency = torch.tensor(sample["adjacency"], device=DEVICE)
    if name == "triple":
        model = GraphSVDD(dim, 32, adjacency).to(DEVICE)
    else:
        model = GraphReconstruction(dim, 48, adjacency, name).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    fit_batches = []
    for case in fit_cases:
        fit_batches.extend(
            [seq for _, seq in _windows(arrays[case], set(range(len(arrays[case]["x"]))), HISTORY)]
        )
    fit = torch.tensor(np.asarray(fit_batches), device=DEVICE)
    if name == "triple":
        with torch.no_grad():
            center = model(fit[:, -1]).mean(dim=0)
        for _ in range(50):
            optimizer.zero_grad()
            loss = ((model(fit[:, -1]) - center) ** 2).mean()
            loss.backward()
            optimizer.step()
    else:
        for _ in range(50):
            optimizer.zero_grad()
            pred = model(fit)
            loss = ((pred - fit[:, -1]) ** 2).mean()
            loss.backward()
            optimizer.step()

    def score(case):
        data = arrays[case]
        sequences = []
        sample_ids = []
        for wi in range(HISTORY - 1, len(data["x"])):
            seq = data["x"][wi - HISTORY + 1 : wi + 1]
            for wj, si, sample_id in data["ids"]:
                if wj == wi:
                    sequences.append(seq)
                    sample_ids.append((si, sample_id))
        scores = []
        with torch.no_grad():
            for start in range(0, len(sequences), 128):
                batch = torch.tensor(np.asarray(sequences[start : start + 128]), device=DEVICE)
                if name == "triple":
                    pred = ((model(batch[:, -1]) - center) ** 2).mean(dim=-1)
                else:
                    pred = ((model(batch) - batch[:, -1]) ** 2).mean(dim=-1)
                scores.extend(
                    pred.detach()
                    .cpu()
                    .numpy()[
                        np.arange(len(pred)), [si for si, _ in sample_ids[start : start + 128]]
                    ]
                )
        return pd.DataFrame({"sample_id": [sid for _, sid in sample_ids], "score": scores})

    calibration_ids = sorted(allowed_calibration_ids)
    cal = pd.concat([score(case) for case in cal_cases], ignore_index=True)
    cal = cal[cal.sample_id.isin(calibration_ids)].set_index("sample_id")
    if cal.index.duplicated().any():
        raise ValueError("Calibration scores contain duplicate frozen IDs")
    missing_calibration = len(set(calibration_ids) - set(cal.index))
    cal = cal.reindex(calibration_ids)
    cal["score"] = cal["score"].fillna(0.0)
    threshold = float(np.quantile(cal.score.to_numpy(), 0.99))
    test = pd.concat([score(case) for case in test_cases], ignore_index=True).set_index("sample_id")
    target_ids = sorted(set(allowed_test_ids))
    missing_history = len(set(target_ids) - set(test.index))
    test = test.reindex(target_ids)
    test["score"] = test["score"].fillna(0.0)
    frame = test.join(labels[["label", "case_id"]], how="inner").dropna(subset=["label"])
    y, s = frame.label.astype(int), frame.score
    per_case = {}
    for case_id, part in frame.groupby("case_id", sort=True):
        case_y = part.label.astype(int).to_numpy()
        case_s = part.score.to_numpy(float)
        has_both = len(np.unique(case_y)) == 2
        per_case[str(case_id)] = {
            "rows": len(part),
            "positive": int((case_y == 1).sum()),
            "negative": int((case_y == 0).sum()),
            "auroc": float(roc_auc_score(case_y, case_s)) if has_both else None,
            "ap": float(average_precision_score(case_y, case_s)) if has_both else None,
        }
    eligible_cases = [
        item for item in per_case.values() if item["auroc"] is not None and item["ap"] is not None
    ]
    return dict(
        name=name,
        seed=seed,
        threshold=threshold,
        rows=len(frame),
        missing_history=missing_history,
        missing_calibration=missing_calibration,
        auroc=roc_auc_score(y, s),
        ap=average_precision_score(y, s),
        per_case=per_case,
        per_case_auroc_macro=(
            float(np.mean([item["auroc"] for item in eligible_cases])) if eligible_cases else None
        ),
        per_case_ap_macro=(
            float(np.mean([item["ap"] for item in eligible_cases])) if eligible_cases else None
        ),
        per_case_cases_with_both_classes=len(eligible_cases),
        recall=float(((s > threshold) & y.eq(1)).sum() / max(1, y.sum())),
        normal_fpr=float(((s > threshold) & y.eq(0)).sum() / max(1, (y == 0).sum())),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    tables, prepared, _ = read_prepared(args.prepared)
    bundle, _ = accepted_bundle(args.bundle, args.acceptance)
    tables, lists = rebind(tables, prepared, bundle)
    fit_case_ids = set(prepared.identity[prepared.identity.sample_id.isin(lists.fit)].case_id)
    arrays, _ = build_arrays(prepared, tables, args.raw_root, fit_case_ids)
    labels = tables.labels.set_index("sample_id")
    case_split = (
        prepared.identity.groupby("case_id")
        .first()
        .join(tables.splits.set_index("sample_id")["split"])
    )
    fit = [
        c
        for c in arrays
        if c in set(prepared.identity[prepared.identity.sample_id.isin(lists.fit)].case_id)
    ]
    cal = [
        c
        for c in arrays
        if c in set(prepared.identity[prepared.identity.sample_id.isin(lists.calibration)].case_id)
    ]
    test = [c for c in arrays if c not in fit and c not in cal]
    for name in ("mstgad", "anofusion", "triple"):
        print(
            json.dumps(
                run_model(
                    name,
                    arrays,
                    fit,
                    cal,
                    test,
                    labels,
                    set(lists.calibration),
                    set(lists.main_test),
                    args.seed,
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
