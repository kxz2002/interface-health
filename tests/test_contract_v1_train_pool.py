import dataclasses
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.contracts.contract_config import ContractConfig

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


_CONTRACT_CONFIG_FIELDS = {f.name for f in dataclasses.fields(ContractConfig)}


def _cfg_with_fractions(
    tmp_path: Path,
    name: str,
    base: str = "configs/contract/v1_expanded_pool.yaml",
    **overrides: float,
) -> str:
    """基于 base（默认 v1_expanded_pool.yaml）产出一份覆写了指定 fraction 的临时 config。

    用 YAML 往返（load → 改 dict → dump）而不是字符串追加：字符串追加会在
    base 已声明同名字段时产生重复 key——PyYAML 静默取最后一个，测试看似能过，
    但一旦有人调整字段顺序或加注释就会悄悄读到另一个值。

    参数名会校验是否为 ContractConfig 的真实字段——loader 用 raw.get(...) 读取，
    未知 key 会被静默忽略、fraction 退化为默认 0.0，导致 fraction=1.0 的"不吸收"
    断言在参数名打错时 vacuously 通过而不报错。
    """
    unknown = set(overrides) - _CONTRACT_CONFIG_FIELDS
    if unknown:
        raise AssertionError(f"未知 ContractConfig 字段: {sorted(unknown)}")
    raw = yaml.safe_load((REPO_ROOT / base).read_text())
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
    """inject 阶段非目标行按 fraction 部分吸收进训练池。

    两个故障 case 各贡献 5 窗 × 1 个非目标 endpoint = 5 行候选，int(5*0.2)=1 窗 →
    每个 case 1 行进 train、4 行留 eval，合计 2 行 train / 8 行 eval。

    entry 025 前这里只有 Lv_E_NTGT_travel 贡献候选（1 行 train / 4 行 eval），因为
    Lv_D_CASELVL_travel 走 case 级 fallback、inject 阶段 is_endpoint_anomaly 恒 True，
    判据 ~is_endpoint_anomaly 选不出任何行。修复后该 case 走 service 档
    （target_service=ts-travel-service），其非目标 service（travel2）的 inject 行不再
    是正样本，因而进入候选池——这正是修复的目的：那些行确实是未受冲击的正常数据。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(tmp_path, "inject_02", fault_inject_nontarget_train_fraction=0.2)
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    absorbed = train[train["source_phase"] == "fault_inject_nontarget"]
    assert (
        len(absorbed) == 2
    ), f"两个 case 各吸收最早 1 窗 × 1 endpoint = 2 行，实际 {len(absorbed)}"
    assert (absorbed["phase"] == "inject").all()
    assert set(absorbed["case_id"]) == {"Lv_E_NTGT_travel", "Lv_D_CASELVL_travel"}
    assert not absorbed["is_endpoint_anomaly"].any()
    # 被吸收的必须都是非目标行：两个 case 的 target 都落在 ts-travel-service 上
    assert (absorbed["service_name"] != "ts-travel-service").all()

    eval_inject_nontarget = eval_all[
        (eval_all["phase"] == "inject") & (~eval_all["is_endpoint_anomaly"])
    ]
    # 18 行 = 两个可观测 case 各留 4 行未吸收（8） + 不可观测 case 的全部 10 行 inject
    # （被 label_target_observable 闸门挡在候选池外，整段留 eval，见
    # test_unobservable_target_inject_rows_never_absorbed）
    assert (
        len(eval_inject_nontarget) == 18
    ), f"应留 8 行未吸收 + 10 行不可观测 inject 行，实际 {len(eval_inject_nontarget)}"


def test_inject_target_rows_never_absorbed_regardless_of_fraction(tmp_path):
    """正样本（inject 阶段命中 target 的行）不管 fraction 多高都不进训练池。

    这是 is_endpoint_anomaly 判据的核心保护。fraction=1.0 让任何判据错误必然暴露：
    - 若 inject 侧误用 is_target_endpoint==False，service 档 case（该列全 NaN）的
      正样本会被当成"非目标"吸收；
    - 若漏掉 label_target_observable 闸门，target 不可观测 case 的全部 inject 行
      （正样本恒为 0）会整段被吸收。

    正样本 10 行（entry 025 前是 15 行）：Lv_E_NTGT_travel 贡献 5（目标 endpoint），
    Lv_D_CASELVL_travel 贡献 5（target_service=ts-travel-service 命中的那个 endpoint）。
    差额 5 行正是修复掉的——CASELVL 的 travel2 行原先被 case 级 fallback 误标为正样本。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(tmp_path, "inject_10", fault_inject_nontarget_train_fraction=1.0)
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    assert not train["is_endpoint_anomaly"].any(), "训练池吸收了正样本"
    positives = eval_all[eval_all["is_endpoint_anomaly"]]
    assert len(positives) == 10, f"正样本应恒为 10 行，实际 {len(positives)}"
    # 正样本必须全部落在 target service 上（两个 case 的 target_service 都是 travel）
    assert (positives["service_name"] == "ts-travel-service").all()
    # 正向对照：证明 fraction 确实被消费了，否则上面的"未被吸收"断言在 fraction
    # 静默退化为 0.0 时会 vacuously 通过——fraction=1.0 应吸收两个 case 各 5 个
    # 非目标 inject 窗口，共 10 行
    assert (train["source_phase"] == "fault_inject_nontarget").sum() == 10


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
    assert (
        len(target_recover_eval) == 5
    ), f"目标 endpoint 的 recover 窗应全留 eval，实际 {len(target_recover_eval)}"
    # 正向对照：证明 fraction 确实被消费了，否则上面的"未被吸收"断言在 fraction
    # 静默退化为 0.0 时会 vacuously 通过——fraction=1.0 应吸收 Lv_E_NTGT_travel
    # 的全部 5 个非目标 recover 窗口进 train
    assert (train["source_phase"] == "fault_recover_nontarget").sum() == 5


def test_non_endpoint_granularity_recover_rows_never_absorbed(tmp_path):
    """label_granularity != 'endpoint' 的 case，其 recover 行不管 fraction 多高都不吸收。

    entry 025 的显式决策：service 档 case 虽然理论上可用 service_name != target_service
    区分目标/非目标，但 recover 侧不纳入——本次只修 inject 侧的标签 bug，多改一处会让
    AUROC 变化无法归因；且 recover 阶段本就是保守排除的（系统未验证回到正常态）。
    实现上靠 `label_granularity == "endpoint"` 这个既有判据天然挡住，无需额外代码。

    这是本轮最容易被静默破坏的行为：若实现漏掉那层过滤，这批 recover 行会全部被当成
    "非目标"吸收进训练池——其中可能包含实际就是故障发生地的 endpoint，污染训练池对
    "正常"的定义。
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
    ).any(), "service 档 case 的 recover 行被吸收了"
    # 该 case 的 10 行 recover（5 窗 × 2 endpoint）必须整段留在 eval_all
    case_recover_eval = eval_all[
        (eval_all["case_id"] == "Lv_D_CASELVL_travel") & (eval_all["phase"] == "recover")
    ]
    assert (
        len(case_recover_eval) == 10
    ), f"service 档 case 的 recover 行应整段留 eval，实际 {len(case_recover_eval)}"
    # 正向对照：证明 fraction 确实被消费了，否则上面的"未被吸收"断言在 fraction
    # 静默退化为 0.0 时会 vacuously 通过——fraction=1.0 应吸收 Lv_E_NTGT_travel
    # 的全部 5 个非目标 recover 窗口进 train
    assert (train["source_phase"] == "fault_recover_nontarget").sum() == 5


def test_label_granularity_derives_from_target_fields_not_anomaly_level(tmp_path):
    """label_granularity 的判据是 case_meta 里有哪一级 target 字段，不是 anomaly_level。

    fixture 里 Lv_D_CASELVL_travel 的 anomaly_level 故意写成 'endpoint' 却没有
    target_endpoint 字段（只有 target_service）。若实现（现在或将来）改用 anomaly_level
    判断，这个 case 会被误判为 endpoint 级精确标签，其正样本判据会错用
    is_target_endpoint（该列在这个 case 上全 NaN），且 recover 行会被错误纳入吸收候选。
    正确结果是 service 档——entry 025 前这里断言的是 'case'，那是二元设计下的旧预期。
    """
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir, config="configs/contract/v1_expanded_pool.yaml", dataset=_NTGT_DATASET)

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    case_rows = eval_all[eval_all["case_id"] == "Lv_D_CASELVL_travel"]
    assert (case_rows["anomaly_level"] == "endpoint").all(), "fixture 前提变了"
    assert (case_rows["label_granularity"] == "service").all()

    ntgt_rows = eval_all[eval_all["case_id"] == "Lv_E_NTGT_travel"]
    assert (ntgt_rows["label_granularity"] == "endpoint").all()


def test_unobservable_target_inject_rows_never_absorbed(tmp_path):
    """target 不可观测的 case（tsdb-mysql 类基础设施组件）的 inject 行不进训练池。

    entry 025 的 label_target_observable 闸门存在的唯一理由。这类 case 正样本恒为 0
    （target_service 不在 endpoint→service 映射的值域里），于是 ~is_endpoint_anomaly
    对其**全部** inject 行恒为 True——少了这道闸，fraction=1.0 会把整段 inject 行吸进
    训练池。而数据库/网关挂掉时 8 个 service 无一不受影响，根本不存在"未受冲击的非目标
    service"，吸收进去等于把故障数据当正常数据喂给 One-Class 模型、污染正常边界。

    评估侧仍保持诚实：这批行留在 eval_all 当负样本，该 case 分层 AUROC 为 null
    （单一类别），与 entry 016/023 对 Lv_S_KILLPOD_gateway 的处理惯例一致。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(
        tmp_path,
        "unobs_all_10",
        fault_inject_nontarget_train_fraction=1.0,
        fault_recover_nontarget_train_fraction=1.0,
    )
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    unobs = "Lv_D_UNOBSERVABLE_mysql"

    # fixture 前提：该 case 确实被判定为不可观测且零正样本
    unobs_eval = eval_all[eval_all["case_id"] == unobs]
    assert not unobs_eval.empty
    assert not unobs_eval["label_target_observable"].any()
    assert int(unobs_eval["is_endpoint_anomaly"].sum()) == 0

    # 核心断言：一行 inject 都没被吸收
    absorbed_inject = train[
        (train["case_id"] == unobs) & (train["source_phase"] == "fault_inject_nontarget")
    ]
    assert (
        len(absorbed_inject) == 0
    ), f"不可观测 case 的 inject 行被吸收了 {len(absorbed_inject)} 行"
    assert (
        not (train["case_id"] == unobs).any()
        or (train[train["case_id"] == unobs]["phase"] != "inject").all()
    )

    # 行守恒：全部 10 行 inject（5 窗 × 2 endpoint）必须留在 eval_all，不能被静默丢弃
    assert (
        int((unobs_eval["phase"] == "inject").sum()) == 10
    ), "不可观测 case 的 inject 行漏出了 eval_all"
    assert set(train["sample_id"]) & set(eval_all["sample_id"]) == set()


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
    # 3 个故障 case × 30 = 102）减去 train_val（Normal 三路切分里既不进 train 也不进
    # eval_all 的那一份，6 窗 × int(6*0.2)=1 窗 × 2 endpoint = 2 行）。
    # entry 025 新增 Lv_D_UNOBSERVABLE_mysql 后故障 case 从 2 个变 3 个，总数 72→102。
    # 这条断言同时守住不可观测 case 的 inject 行不被静默丢弃——它们既不进训练池、
    # 又不是正样本，若 eval_all 的分段拼接漏收这一段，行数就会短 10 行。
    assert (
        len(train) + len(eval_all) == 102 - 2
    ), f"三路全吸收后行数不守恒：train={len(train)} eval_all={len(eval_all)}"


def test_nontarget_fractions_are_noop_when_expand_train_pool_false(tmp_path):
    """两个新 fraction 只在 expand_train_pool=true 时被消费。expand_train_pool=false
    时即便都设成 1.0，train.parquet 也必须恒等于 train_fit.parquet、不含 source_phase
    列——照抄 fault_baseline_train_fraction 的既有约定（见该字段的 noop 测试）。
    """
    cfg = _cfg_with_fractions(
        tmp_path,
        "nontarget_noop",
        base="configs/contract/v1.yaml",
        fault_inject_nontarget_train_fraction=1.0,
        fault_recover_nontarget_train_fraction=1.0,
    )

    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    train_fit = pd.read_parquet(out_dir / "train_fit.parquet")
    assert set(train["sample_id"]) == set(train_fit["sample_id"])
    assert "source_phase" not in train.columns

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    # 故障 case 的 inject/recover 行必须全留 eval（3 个故障 case × 5 窗 × 2 endpoint
    # = 30 行each；entry 025 新增第 3 个 fixture case 后从 20 涨到 30）
    assert (eval_all["phase"] == "inject").sum() == 30
    assert (eval_all["phase"] == "recover").sum() == 30
