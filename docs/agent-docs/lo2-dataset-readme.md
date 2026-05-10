# LO2 Dataset README

来源: [LO2: Microservice Dataset of Logs and Metrics (Zenodo)](https://zenodo.org/records/14938118)

**DOI**: 10.5281/zenodo.14938118 | **Version**: v3 (2025-02-28) | **License**: CC BY 4.0

## 数据集概述

LO2 是一个微服务日志与指标数据集，完整解压后约 540 GB。样本数据（`lo2-sample.zip`，1.1 GB）已解压至 `data/raw/lo2-sample/`。

## 目录结构

```
data/raw/lo2-sample/
├── metrics/                      # 时间序列指标 (100 个 CSV)
│   ├── light-oauth2-data-<timestamp>.csv
│   └── ...
└── logs/                         # 非结构化日志 (100 个子目录)
    └── light-oauth2-data-<timestamp>/
        ├── correct/             # 正常请求（基线类）
        ├── access_token_401/
        ├── access_token_404/
        └── ... (共 55 个子目录)
```

**每个 timestamp 对应一个 Run**，metrics CSV 与 logs 子目录一一配对。

---

## Metrics 数据

| 属性 | 值 |
|------|---|
| 单个 CSV 形状 | 172 行 × 1127 列 |
| 时间范围 | ~800 秒/Run |
| 采样频率 | ~5 秒/条 |
| test_name 类别 | 55 种 |

**指标列分类**:

| 前缀 | 数量 | 来源 |
|------|------|------|
| `go_*` | 33 | Go 运行时（gc、goroutines、memstats...）|
| `node_*` | 1073 | node_exporter 系统指标（CPU、内存、磁盘、网络...）|

---

## Logs 数据

### 55 个错误类型子目录

按 HTTP 方法分组：

**GET**: `get_client_404_no_client`, `get_client_page_400_no_page`, `get_service_404_no_service`, `get_service_page_400_no_page`, `get_token_404`, `get_token_page_400_no_page`, `get_user_404_no_user`, `get_user_page_400_no_page`

**POST (access_token)**: `access_token_auth_header_error_401`, `access_token_authorization_form_401`, `access_token_client_id_not_found_404`, `access_token_client_secret_wrong_401`, `access_token_form_urlencoded_400`, `access_token_illegal_grant_type_400`, `access_token_missing_authorization_header_400`

**POST (authorization_code)**: `authorization_code_client_id_missing_400`, `authorization_code_invalid_client_id_404`, `authorization_code_invalid_password_401`, `authorization_code_missing_response_type_400`, `authorization_code_response_not_code_400`

**POST (register)**: `register_client_400_clientProfile`, `register_client_400_clientType`, `register_client_404_no_user`, `register_service_400_service_id`, `register_service_400_service_type`, `register_service_404_no_user`, `register_user_400_email_exists`, `register_user_400_no_password`, `register_user_400_password_no_match`, `register_user_400_user_exists`

**POST (PKCE)**: `code_challenge_invalid_format_pkce_400`, `code_challenge_too_long_pkce_400`, `code_challenge_too_short_pkce_400`, `code_verifier_missing_pkce_400`, `code_verifier_too_long_pkce_400`, `code_verifier_too_short_pkce_400`, `invalid_code_challenge_method_pkce_400`, `invalid_code_verifier_format_PKCE_400`, `verification_failed_pkce_400`

**PUT (update)**: `update_client_400_clientProfile`, `update_client_400_clientType`, `update_client_404_clientId`, `update_client_404_ownerId`, `update_password_400_not_match`, `update_password_401_wrong_password`, `update_password_404_user_not_found`, `update_service_404_service_id`, `update_service_404_user_id`, `update_user_404_no_user`

**DELETE**: `delete_client_404_no_client`, `delete_service_404_no_service`, `delete_token_404`, `delete_user_404_no_user`

**Baseline**: `correct`（正常请求，日志表示无错误发生）

### 7 个微服务日志文件

每个错误类型子目录下有 7 个服务日志文件：

| 服务名 | 说明 |
|--------|------|
| `light-oauth2-oauth2-client-1.log` | OAuth2 客户端服务 |
| `light-oauth2-oauth2-code-1.log` | 授权码服务 |
| `light-oauth2-oauth2-key-1.log` | Key 服务 |
| `light-oauth2-oauth2-refresh-token-1.log` | 刷新令牌服务 |
| `light-oauth2-oauth2-service-1.log` | 主服务 |
| `light-oauth2-oauth2-token-1.log` | Token 服务 |
| `light-oauth2-oauth2-user-1.log` | 用户服务 |

**每个 Run 总日志文件数**: 55 × 7 = 385 个

### 日志格式

Java stack trace，示例：

```
at com.networknt.handler.Handler.next(Handler.java:233)
at com.networknt.traceability.TraceabilityHandler.handleRequest(TraceabilityHandler.java:68)
at io.undertow.server.HttpServerExchange$1.run(HttpServerExchange.java:841)
```

---

## 数据对应关系

```
metrics/light-oauth2-data-<ts>.csv   ←→   logs/light-oauth2-data-<ts>/
```

- metrics 中每行有一个 `test_name` 字段（55 种值）
- 对应的 logs 子目录下有同名的错误类型子目录
- 日志文件内容为 Java stack trace，按服务分离

---

## 使用建议

1. **时序对齐**: 通过 `<timestamp>` + `test_name` 匹配 metrics 和 logs
2. **日志解析**: 可参考 `scripts/lo2-scripts/reduce_logs.py` 去除初始化行，`logstats.py` 统计日志行数
3. **指标选择**: node_exporter 指标（1073 列）可能存在较多冗余，可通过 PCA 或特征选择降维
4. **异常检测任务**: `test_name` 作为分类标签，54 种错误类型 + 1 种 correct

---

## 相关文档

- [lo2-scripts](lo2-scripts.md) — 数据处理脚本说明
