import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).parents[1]


def _run_v1_build(
    out_dir: Path,
    config: str = "configs/contract/v1_expanded_pool.yaml",
    dataset: str = "tests/fixtures/split_fraction_mini.yaml",
) -> None:
    # 这些测试验证训练池扩容 + baseline 时序切分行为，故默认走 v1_expanded_pool.yaml
    # （expand_train_pool=true）+ split_fraction_mini（故障 case 有 5 个 baseline 窗口，
    # int(5*0.2)=1，能验证非零分数切分）。mini_dataset.yaml 的故障 case 每个只有
    # 1 个 baseline 窗口，int(1*0.2)=0，无法验证分数切分，不能用于这批测试。
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / config),
            "--dataset",
            str(REPO_ROOT / dataset),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def _cfg_with_fractions(tmp_path: Path, name: str, **overrides: float) -> str:
    """基于 v1_expanded_pool.yaml 产出一份覆写了指定 fraction 的临时 config。

    用 YAML 往返（load → 改 dict → dump）而不是字符串追加：字符串追加会在
    v1_expanded_pool.yaml 已声明同名字段时产生重复 key——PyYAML 静默取最后一个，
    测试看似能过，但一旦有人调整字段顺序或加注释就会悄悄读到另一个值。
    """
    raw = yaml.safe_load((REPO_ROOT / "configs/contract/v1_expanded_pool.yaml").read_text())
    raw.update(overrides)
    cfg_path = tmp_path / f"{name}.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
    return str(cfg_path)


def test_train_pool_includes_fault_baseline_rows(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)

    train = pd.read_parquet(out_dir / "train.parquet")
    # mini fixture 含故障 case（如 Lv_E_SPLITTEST_travel），其 baseline 阶段行
    # 现在应出现在 train.parquet 里（扩容前 train 只含 Normal case）。
    assert (
        train["anomaly_type"] != "Normal"
    ).any(), "train.parquet 未吸收任何故障 case 的 baseline 行，训练池扩容未生效"
    # 吸收的行必须确实是 baseline 阶段，不能混入 inject/recover
    non_normal = train[train["anomaly_type"] != "Normal"]
    assert (non_normal["phase"] == "baseline").all()


def test_train_pool_excludes_inject_and_recover_rows(tmp_path):
    """split_fraction_mini 上 train 不含 inject/recover 行。

    注意这**不是**全局不变量：该 fixture 的故障 case 只有 1 个 endpoint 且它就是
    目标 endpoint，故 inject 非目标候选池（is_endpoint_anomaly==False）与 recover
    非目标候选池（is_target_endpoint==False）恒为空，两个 nontarget fraction 取
    任何值都吸收不到行。在有 endpoint 级 fan-out 的 fixture 上（见
    tests/fixtures/nontarget_split_mini），train 合法地会含 inject/recover 行——
    那批行由 test_inject_nontarget_rows_partially_absorbed_into_train 等测试覆盖。
    """
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    assert not (train["phase"] == "recover").any()
    assert not (train["phase"] == "inject").any()


def test_source_phase_column_present_and_correct(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    assert "source_phase" in train.columns
    normal_rows = train[train["anomaly_type"] == "Normal"]
    fault_rows = train[train["anomaly_type"] != "Normal"]
    assert (normal_rows["source_phase"] == "normal_case").all()
    # mini fixture 含故障 case，fault_rows 不应为空——若为空说明训练池扩容未生效，
    # 这里必须硬断言，不能因为 len(fault_rows)==0 而静默跳过下面的校验
    assert len(fault_rows) > 0
    assert (fault_rows["source_phase"] == "fault_baseline").all()


def test_eval_all_and_train_pool_sample_id_disjoint_no_leakage(tmp_path):
    """核心防泄漏不变量（issue #16 修复后仍必须成立且加强）：train_pool 与 eval_all
    的 sample_id 必须互斥。issue #16 前的实现靠"baseline 整段摘出 eval"来保证互斥；
    修复后 baseline 按时间窗时序切分，一部分进 train、其余留 eval，互斥性改由
    split_fault_phase_temporal 的整窗切分保证（同一窗不会既在 train 又在 eval）。
    这条断言是唯一的硬约束，语义比"eval 不含 baseline 行"更本质。"""
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    assert set(train["sample_id"]) & set(eval_all["sample_id"]) == set()


def test_eval_all_retains_majority_of_fault_baseline_rows(tmp_path):
    """issue #16 修复的正向断言：故障 baseline 行不再被整段摘出 eval，大部分（新 fixture
    里 5 窗中的 4 窗 = 80%）应留在 eval_all 维持负样本类别平衡。train 只吸收最早 1 窗。"""
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    train_baseline = train[(train["anomaly_type"] != "Normal") & (train["phase"] == "baseline")]
    eval_baseline = eval_all[
        (eval_all["anomaly_type"] != "Normal") & (eval_all["phase"] == "baseline")
    ]
    train_windows = train_baseline["timestamp_window_ms"].nunique()
    eval_windows = eval_baseline["timestamp_window_ms"].nunique()
    assert train_windows == 1, "fraction=0.2 × 5 窗应吸收最早 1 窗进 train"
    assert eval_windows == 4, "其余 4 窗 baseline 应留在 eval_all"


def test_default_v1_config_does_not_expand_train_pool(tmp_path):
    """configs/contract/v1.yaml 默认 expand_train_pool=false（本次新增开关的
    默认值），train.parquet 必须与 train_fit.parquet 完全一致（不吸收故障
    baseline 行），eval_all 必须含故障 case 的全部阶段（baseline 不摘除）。
    这是与 entry 012 既有实验数字保持可比的行为，必须锁住不被悄悄改回扩容。
    """
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir, config="configs/contract/v1.yaml")

    train = pd.read_parquet(out_dir / "train.parquet")
    train_fit = pd.read_parquet(out_dir / "train_fit.parquet")
    assert set(train["sample_id"]) == set(train_fit["sample_id"])
    assert "source_phase" not in train.columns

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    # mini fixture 含故障 case，baseline 阶段行必须仍留在 eval_all 里（未被摘除）
    fault_baseline_in_eval = eval_all[
        (eval_all["anomaly_type"] != "Normal") & (eval_all["phase"] == "baseline")
    ]
    assert len(fault_baseline_in_eval) > 0


def test_fault_baseline_train_fraction_is_noop_when_expand_train_pool_false(tmp_path):
    """fault_baseline_train_fraction 只在 expand_train_pool=true 时被 _write_v1 消费
    （见 contract_config.py 里该字段的注释）。这条测试把该假设钉死为可执行断言：
    expand_train_pool=false 时，即便 fraction 显式设成远离默认值 0.2 的 0.9，
    输出也必须与 v1.yaml（默认 fraction）完全一致——train.parquet 恒等于
    train_fit.parquet，不吸收任何故障 baseline 行。若未来重构 _write_v1 把两个
    分支合并，这条测试能防止 fraction 在 expand_train_pool=false 时被意外消费。"""
    cfg_path = tmp_path / "v1_noop_check.yaml"
    cfg_path.write_text(
        (REPO_ROOT / "configs/contract/v1.yaml").read_text()
        + "fault_baseline_train_fraction: 0.9\n"
    )

    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir, config=str(cfg_path))

    train = pd.read_parquet(out_dir / "train.parquet")
    train_fit = pd.read_parquet(out_dir / "train_fit.parquet")
    assert set(train["sample_id"]) == set(train_fit["sample_id"])
    assert "source_phase" not in train.columns

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    fault_baseline_in_eval = eval_all[
        (eval_all["anomaly_type"] != "Normal") & (eval_all["phase"] == "baseline")
    ]
    assert (
        len(fault_baseline_in_eval) > 0
    ), "fraction=0.9 若被意外消费，会把大部分 baseline 吸收进 train"


def test_contract_v1_838_variant_is_pure_normal(tmp_path):
    """2x2 归因实验依赖的手工构造 contract_v1_838：确认 train.parquet 替换为
    train_fit.parquet 后确实是纯 Normal（无 source_phase 列或全为 normal_case），
    防止归因实验的"原始838行"对照组混入扩容数据。"""
    out_dir = tmp_path / "contract_v1"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v1.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    train_fit = pd.read_parquet(out_dir / "train_fit.parquet")
    assert (train_fit["anomaly_type"] == "Normal").all()


_NTGT_DATASET = "tests/fixtures/nontarget_split_mini.yaml"


def test_inject_nontarget_rows_partially_absorbed_into_train(tmp_path):
    """inject 阶段非目标 endpoint 行按 fraction 部分吸收进训练池。

    nontarget_split_mini 的 Lv_E_NTGT_travel 有 5 个 inject 窗口 × 1 个非目标
    endpoint = 5 行候选，int(5*0.2)=1 窗 → 1 行进 train、4 行留 eval。
    Lv_D_CASELVL_travel（case 级标签）在 inject 阶段所有行 is_endpoint_anomaly
    都 fallback 为 True，不进候选池，故总候选恒为 5 行而非 10 行。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(tmp_path, "inject_02", fault_inject_nontarget_train_fraction=0.2)
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    absorbed = train[train["source_phase"] == "fault_inject_nontarget"]
    assert len(absorbed) == 1, f"应吸收最早 1 窗 × 1 endpoint = 1 行，实际 {len(absorbed)}"
    assert (absorbed["phase"] == "inject").all()
    assert (absorbed["case_id"] == "Lv_E_NTGT_travel").all()
    assert not absorbed["is_endpoint_anomaly"].any()

    eval_inject_nontarget = eval_all[
        (eval_all["phase"] == "inject") & (~eval_all["is_endpoint_anomaly"])
    ]
    # 剩 4 行来自 Lv_E_NTGT_travel，另 0 行来自 case 级 case（其 inject 行全是正样本）
    assert len(eval_inject_nontarget) == 4, len(eval_inject_nontarget)


def test_inject_target_rows_never_absorbed_regardless_of_fraction(tmp_path):
    """正样本（inject 阶段目标 endpoint 行）不管 fraction 多高都不进训练池。

    这是 is_endpoint_anomaly 判据的核心保护：若实现误用 is_target_endpoint==False
    作为 inject 侧判据，case 级标签 case（is_target_endpoint 全 False）的全部
    inject 行都会被当成"非目标"吸收——而它们的 is_endpoint_anomaly 是 True，
    就是正样本。fraction=1.0 让这个 bug 必然暴露。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(tmp_path, "inject_10", fault_inject_nontarget_train_fraction=1.0)
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    assert not train["is_endpoint_anomaly"].any(), "训练池吸收了正样本"
    # 两个故障 case 各 5 窗 inject：endpoint 级 case 贡献 5 个正样本（目标 endpoint），
    # case 级 case 的 10 行 inject 全是正样本（fallback），合计 15 行必须全留 eval
    positives = eval_all[eval_all["is_endpoint_anomaly"]]
    assert len(positives) == 15, f"正样本应恒为 15 行，实际 {len(positives)}"


def test_recover_nontarget_rows_partially_absorbed_into_train(tmp_path):
    """recover 阶段非目标 endpoint 行按独立 fraction 部分吸收进训练池。

    只有 endpoint 级标签 case 的非目标 endpoint 参与：5 窗 × 1 endpoint = 5 行
    候选，int(5*0.2)=1 窗 → 1 行进 train。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(tmp_path, "recover_02", fault_recover_nontarget_train_fraction=0.2)
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    absorbed = train[train["source_phase"] == "fault_recover_nontarget"]
    assert len(absorbed) == 1, f"应吸收最早 1 窗 × 1 endpoint = 1 行，实际 {len(absorbed)}"
    assert (absorbed["phase"] == "recover").all()
    assert (absorbed["case_id"] == "Lv_E_NTGT_travel").all()
    assert not absorbed["is_target_endpoint"].any()


def test_recover_target_endpoint_rows_never_absorbed(tmp_path):
    """endpoint 级标签 case 的目标 endpoint recover 行永不被吸收——沿用 entry 014
    的保守排除理由（系统未稳定回正常态，分布未验证）。fraction=1.0 时必须仍成立。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(tmp_path, "recover_10", fault_recover_nontarget_train_fraction=1.0)
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    recover_in_train = train[train["phase"] == "recover"]
    assert not recover_in_train["is_target_endpoint"].any(), "目标 endpoint 的 recover 行被吸收了"
    # Lv_E_NTGT_travel 目标 endpoint 的 5 个 recover 窗必须全留 eval
    target_recover_eval = eval_all[
        (eval_all["phase"] == "recover")
        & (eval_all["case_id"] == "Lv_E_NTGT_travel")
        & eval_all["is_target_endpoint"]
    ]
    assert len(target_recover_eval) == 5, len(target_recover_eval)


def test_case_level_label_recover_rows_never_absorbed(tmp_path):
    """label_granularity=='case' 的 case，其 recover 行不管 fraction 多高都不被吸收。

    这是本轮最容易被静默破坏的行为：case 级标签 case 没有 target_endpoint 字段、
    is_target_endpoint 全为 False，若实现漏掉 label_granularity=='endpoint' 这层
    过滤，这批 recover 行会全部被当成"非目标"吸收进训练池——其中可能包含实际就是
    故障发生地的 endpoint，污染训练池对"正常"的定义。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(
        tmp_path, "recover_case_10", fault_recover_nontarget_train_fraction=1.0
    )
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    assert not (
        (train["case_id"] == "Lv_D_CASELVL_travel") & (train["phase"] == "recover")
    ).any(), "case 级标签 case 的 recover 行被吸收了"
    # 该 case 的 10 行 recover（5 窗 × 2 endpoint）必须整段留在 eval_all
    case_recover_eval = eval_all[
        (eval_all["case_id"] == "Lv_D_CASELVL_travel") & (eval_all["phase"] == "recover")
    ]
    assert len(case_recover_eval) == 10, len(case_recover_eval)


def test_label_granularity_derives_from_target_endpoint_not_anomaly_level(tmp_path):
    """label_granularity 的判据是 target_endpoint 字段是否存在，不是 anomaly_level。

    fixture 里 Lv_D_CASELVL_travel 的 anomaly_level 故意写成 'endpoint' 却没有
    target_endpoint 字段。若实现（现在或将来）改用 anomaly_level 判断，这个 case
    会被误判为 endpoint 级精确标签，其 recover 行会被错误纳入吸收候选。
    """
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir, config="configs/contract/v1_expanded_pool.yaml", dataset=_NTGT_DATASET)

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    case_rows = eval_all[eval_all["case_id"] == "Lv_D_CASELVL_travel"]
    assert (case_rows["anomaly_level"] == "endpoint").all(), "fixture 前提变了"
    assert (case_rows["label_granularity"] == "case").all()

    ntgt_rows = eval_all[eval_all["case_id"] == "Lv_E_NTGT_travel"]
    assert (ntgt_rows["label_granularity"] == "endpoint").all()


def test_all_three_fractions_together_keep_sample_id_disjoint(tmp_path):
    """三路吸收（baseline + inject 非目标 + recover 非目标）全部推到 1.0 时，
    train 与 eval_all 的 sample_id 仍必须互斥——这是 Contract v1 存在的理由，
    也是本轮唯一的硬约束。整窗切分天然保证它，本测试防止未来重构破坏。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(
        tmp_path,
        "all_10",
        fault_baseline_train_fraction=1.0,
        fault_inject_nontarget_train_fraction=1.0,
        fault_recover_nontarget_train_fraction=1.0,
    )
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    assert set(train["sample_id"]) & set(eval_all["sample_id"]) == set()
    # 无行丢失：三路全吸收后，两侧行数之和必须等于 fixture 总行数（12 Normal +
    # 30 + 30 = 72）减去 train_val（Normal 三路切分里既不进 train 也不进 eval_all
    # 的那一份，6 窗 × int(6*0.2)=1 窗 × 2 endpoint = 2 行）
    assert len(train) + len(eval_all) == 72 - 2


def test_nontarget_fractions_are_noop_when_expand_train_pool_false(tmp_path):
    """两个新 fraction 只在 expand_train_pool=true 时被消费。expand_train_pool=false
    时即便都设成 1.0，train.parquet 也必须恒等于 train_fit.parquet、不含 source_phase
    列——照抄 fault_baseline_train_fraction 的既有约定（见该字段的 noop 测试）。
    """
    raw = yaml.safe_load((REPO_ROOT / "configs/contract/v1.yaml").read_text())
    raw["fault_inject_nontarget_train_fraction"] = 1.0
    raw["fault_recover_nontarget_train_fraction"] = 1.0
    cfg_path = tmp_path / "v1_nontarget_noop.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))

    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir, config=str(cfg_path), dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    train_fit = pd.read_parquet(out_dir / "train_fit.parquet")
    assert set(train["sample_id"]) == set(train_fit["sample_id"])
    assert "source_phase" not in train.columns

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    # 故障 case 的 inject/recover 行必须全留 eval（各 20 行）
    assert (eval_all["phase"] == "inject").sum() == 20
    assert (eval_all["phase"] == "recover").sum() == 20
