# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [0.3.0] - 2026-10-09

### Added
- `SentenceTransformerEmbedder`（可选依赖 `kb-agent[local]`）与 `--embedding-provider local`：本地语义向量，无需 API Key
- `eval/run_eval.py`：`--reranker {noop,heuristic,cross-encoder}`、`--rerankers` 网格对比、逐条延迟与 P50/P95 统计
- 离线向量 fixture `eval/fixtures/bge-small-zh-v1.5.npz`（0.55 MB）：`--embedding-cache` 无模型、无 Key 复现语义向量列
- 四象限报告 `eval/report-reranker-quadrant.md`（重排器 × 稠密分支质量）

### Changed
- README 增补"重排器 × 稠密分支质量"章节；cross-encoder TODO 标记完成

## [0.2.0] - 2026-10-08

### Fixed
- `KB_MIN_CONFIDENCE` 的 `from_env` 默认值与 dataclass/文档不一致（0.35 vs 0.45）：统一为 `DEFAULT_MIN_CONFIDENCE` 并加回归测试
- 切到远端语义向量后，加载索引会误判"维度不一致"并触发全量重建（重复调用 embedding API）：`KB_EMBEDDING_DIM` 默认改为 `auto`，按向量文件自适应

### Added
- 接口鉴权（`KB_API_KEY`，默认关闭）与并发闸门（`KB_MAX_CONCURRENCY` / `KB_QUEUE_TIMEOUT_SECONDS`）
- 语义缓存（默认关闭）与误命中风险评估脚本 `eval/run_cache_risk.py`
- 评测集 50 → 150 条（120 可回答 + 30 负样本，含 dev/test 划分）
- 切分策略消融（`--strategies`）与离线向量 fixture（`--embedding-cache`、`scripts/make_embedding_fixture.py`）
- 压测离线基线写入 README；`/ask` 返回 `cache_kind`，`/stats` 暴露语义缓存指标

### Changed
- 评测主表与四条结论全部按 150 条新集更新；切分策略的观察标注了"结论依赖指标口径"
- CI 增加 ruff 与覆盖率门槛（78%），当前覆盖率 82%

## [0.1.0] - 2026-09-14

- 初始版本：混合检索（BM25 + 向量 + RRF）、启发式重排、置信度门控、可复现评测与 55 个单元测试
