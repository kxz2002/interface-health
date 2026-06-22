import logging
from pathlib import Path
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """MLflow 实验追踪的轻量封装。

    训练代码只依赖此类，不直接 import mlflow——未来若切换到 W&B 或
    TensorBoard，只需改这里的实现，训练代码零改动。

    用法：
        exp_logger = ExperimentLogger(experiment_name="anofusion-ablation")
        with exp_logger.start_run(run_name="deep_svdd-lr1e-3"):
            exp_logger.log_params({"lr": 1e-3, "hidden_dim": 128})
            exp_logger.log_metrics({"train_loss": 0.42}, step=1)
            exp_logger.log_artifact("outputs/config.yaml")
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "outputs/mlruns",
    ) -> None:
        Path(tracking_uri).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._experiment_name = experiment_name

    def start_run(self, run_name: str | None = None, tags: dict[str, Any] | None = None):
        """返回 mlflow.ActiveRun context manager，供 with 语句使用。"""
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        """将 dict（如 Hydra config snapshot）直接作为 artifact 记录。"""
        mlflow.log_dict(dictionary, artifact_file)
