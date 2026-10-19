# T4 评估执行代码快照

首次提交 hook 的 isort 仅调整 CLI 导入排序，发生在实际 evaluate 完成之后。
这里保存实际执行的 scripts/service_experiment.py 原文（以 .py.txt 防止格式化工具改变快照），其 SHA256 与 evaluation/source_manifest.json
中该路径的 executed_code SHA 一致。evaluation 原始产物与清单未改写，最终仓库 CLI 经 isort 格式化。
原始 sources/outputs 校验完成于格式化之前；之后 CLI 当前文件SHA有此已记录差异。
