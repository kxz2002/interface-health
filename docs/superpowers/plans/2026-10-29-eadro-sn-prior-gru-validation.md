# Eadro SN Prior GRU Cross-Dataset Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 Eadro SocialNetwork 数据上按本项目自己的 Service 先验 GRU、Normal-only fit 和独立 Normal calibration 协议跑通可审计的跨数据集验证。

**Architecture:** 复用现有 SN archive/window adapter 和 `scripts/service_model.py` / `scripts/service_temporal_model.py` 的 GRU 实现。SN 资源字段保留数据集自身语义，训练输入通过 modality mask 进入同一时序接口；Normal reference 只用于 fit/calibration，Fault 只用于诊断性测试，Unknown 不伪造为负类。第一轮先用 seed 42 跑通，再固定协议扩展 seed 1/2。

**Tech Stack:** Python 3.12, pandas/pyarrow, PyTorch, existing service modeling modules, pytest.

**Spec:** `docs/research/service-20261019/cross-dataset-validation.md`、`docs/research/service-20261019/preparation-implementation.md`、`AGENTS.md` 当前 Service 协议。

## Global Constraints

- 原始 ZIP 只读，必须校验 78,753,809 bytes 和 MD5 `4c523faf64770b1e9a709cdd610e07c4`。
- 不复现 Eadro 原论文模型；目标是本项目自己的先验 GRU。
- SN 的 exploratory 时间/单位假设必须写入 manifest，不得冒称已证语义。
- fit 只用 SN 固定 Normal fit 运行，calibration 只用独立 Normal calibration 运行，Fault 只进测试/诊断。
- Unknown 不填 0；没有独立 Service 金标时，主 AUROC/AP 只能输出 null 或明确标为 controlled-exposure proxy。
- 输出写入新目录，禁止覆盖已有产物；每次保存 config、代码/数据 hash、ID 清单、逐样本分数和质量报告。

## Review Focus

- SN 三个 Normal 运行的角色是否与固定 allowlist 一致；测试由 `test_sn_roles_are_bound` 锁定。
- 资源字段缺失或单位未知是否只通过 mask 进入模型；测试 `test_sn_unknown_resource_semantics_are_not_relabelled`。
- 时间历史是否跨运行/服务/划分边界；测试 `test_sn_temporal_context_does_not_cross_boundaries`。
- Unknown/controlled exposure 是否被错误混入 fit/calibration；测试 `test_sn_model_lists_exclude_unknown_from_fit_and_calibration`。
- 输出 manifest 是否绑定原始 ZIP、preparation、label/split 和模型配置；测试 `test_sn_prior_gru_manifest_binds_inputs`。

### Task 1: 生成并验收 SN preparation 产物

**Files:**
- Modify: `scripts/prepare_service_data.py` only if current output binding cannot consume `data/external/eadro_sn/SN Dataset.zip`.
- Test: existing `tests/test_service_preparation.py`, `tests/test_service_label_splits.py`.
- Output: new `outputs/service/sn-preparation-<run_id>/`.

**Interfaces:**
- Consumes: verified archive, `outputs/service/preparation-v1/inventory.json`, `configs/service/sn_exploratory_v1.json`.
- Produces: full-grid SN resource/execution/log/labels/splits tables and manifest-bound preparation path.

- [ ] Run the production preparation command with a fresh output directory.
- [ ] Run the preparation acceptance command and confirm source/archive hashes, fixed run roles, full grid and masks.
- [ ] Record exact output path and quality report for later tasks.

### Task 2: Add SN model-prepared bridge and proxy label boundary

**Files:**
- Create/modify: `src/service/modeling/sn_adapter.py` (SN-specific field mapping and quality masks).
- Modify: `scripts/service_model.py` only to accept an explicit SN preparation profile without changing TT defaults.
- Test: `tests/test_service_sn_model_adapter.py`.

**Interfaces:**
- Consumes: Task 1 preparation tables.
- Produces: `InputTables`/`PreparedData` compatible with `temporal_inputs()`; explicit `label_source`, `label_scope`, and proxy-evaluation membership.

- [ ] Write failing tests for role binding, mask preservation, and no cross-boundary history.
- [ ] Implement the adapter using pure functions; do not rename SN fields to TT semantics.
- [ ] Run focused adapter tests and a synthetic round-trip.

### Task 3: Run prior GRU minimal closed loop

**Files:**
- Create: `configs/service/sn_temporal_prior_gpu_v1.json`.
- Modify: `scripts/service_temporal_model.py` only for profile/config plumbing if required.
- Output: new `outputs/service/sn-prior-gru-seed42-<run_id>/`.

**Interfaces:**
- Consumes: Task 2 model-prepared artifact and fixed config.
- Produces: checkpoint, calibration scores, test/diagnostic scores, thresholds, evaluation report, manifest.

- [ ] Freeze context=4, hidden_dim=32, 50 epochs, seed=42, q99 calibration and existing prior-logit integration.
- [ ] Run fit, score and evaluate without using Fault scores for tuning.
- [ ] Verify checkpoint fit IDs, calibration IDs, no Unknown in metrics, and complete score coverage.

### Task 4: Fixed-seed replication and audit

**Files:**
- Create: `scripts/audit_sn_prior_gru.py` and `tests/test_sn_prior_gru_audit.py`.
- Output: seed 1/2 reports and aggregate markdown/json under `outputs/service/` or research backup.

- [ ] Run identical config for seeds 1 and 2.
- [ ] Recompute metrics from immutable scores and compare manifests.
- [ ] Report AUROC/AP only for evidence-supported proxy scope; report coverage, FPR and Unknown separately.

### Task 5: Optional cross-domain frozen-weight probe

**Files:**
- Add a separate config/report only after Task 3–4 pass.

- [ ] Freeze TT checkpoint and apply only an explicitly defined feature/normalization bridge to SN.
- [ ] Treat this as a diagnostic transfer experiment, never mix it with SN-internal training results.
