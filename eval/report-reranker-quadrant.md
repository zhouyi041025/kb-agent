# 重排器 × 稠密分支质量：四象限报告

- 生成时间：2026-10-09
- 评测集：150 条查询（120 可回答 + 30 负样本），top_k=5，拒答阈值 0.45
- 检索：混合检索 RRF（BM25 + 稠密分支），候选 20 条进入重排
- 重排器三档：`noop`（不重排）/ `heuristic`（启发式，词法信号）/ `cross-encoder`（BAAI/bge-reranker-base，CPU）

## 结果

| 稠密分支 | 重排器 | R@1 | MRR | 引用准确率 | P50 延迟 | P95 延迟 |
| --- | --- | --- | --- | --- | --- | --- |
| 哈希向量 4096（词法） | noop | 0.908 | 0.942 | 0.769 | 0.6 ms | 0.7 ms |
| 哈希向量 4096（词法） | heuristic | 0.925 | 0.953 | 0.773 | 2.1 ms | 2.5 ms |
| 哈希向量 4096（词法） | **cross-encoder** | **0.967** | **0.976** | **0.803** | 899.5 ms | 2794.7 ms |
| 本地语义向量 bge-small-zh 512 | noop | 0.908 | 0.947 | 0.782 | 7.2 ms | 9.7 ms |
| 本地语义向量 bge-small-zh 512 | heuristic | 0.933 | 0.959 | 0.778 | 8.2 ms | 10.3 ms |
| 本地语义向量 bge-small-zh 512 | **cross-encoder** | **0.967** | **0.979** | **0.804** | 930.9 ms | 1101.5 ms |

## 四个观察

1. **cross-encoder 是唯一"换到哪一列都稳定"的收益**：两种稠密分支下 R@1 都是 0.967（比各自 noop 高 0.059），MRR 0.976 / 0.979。代价是单题 P50 约 0.9 秒、P95 1.1–2.8 秒（CPU、20 候选/题），比启发式重排慢两个数量级。
2. **启发式重排在两列都是正收益，但更小**：哈希 +0.017 R@1，语义 +0.025 R@1。它用词法信号（idf 加权覆盖 + 最长公共子串），便宜到几乎免费（P50 2–8 ms）。
3. **旧结论的适用边界被这次复测说清楚了**：更早（50 条集、智谱 embedding-3、2048 维）观察到"启发式重排在语义向量下变成 −0.024"。合起来看：启发式重排的增益随稠密分支变强而收窄、可能翻负（强分支下词法信号开始打乱已排序结果）；本次的 bge-small-zh（512 维、更小）还没到那个拐点。cross-encoder 不受这个拐点影响——它用的是查询-文档交互打分，不是词法覆盖。
4. **稠密分支换成语义向量还会顺带改善拒答**：同一套门控下，混合 RRF 的拒答正确率从 0.800（哈希）升到 0.933（语义），误答率从 0.200 降到 0.067。

## 复现命令

```bash
# 完整重跑（需要 sentence-transformers 与两个本地模型）
python eval/run_eval.py --embedding-provider local --embedding-model BAAI/bge-small-zh-v1.5 \
  --rerankers noop,heuristic,cross-encoder --reranker-model BAAI/bge-reranker-base

# 完全离线复现语义向量列（仓库自带 fixture，0.5 MB，不需要下载模型/Key）
python eval/run_eval.py --embedding-cache eval/fixtures/bge-small-zh-v1.5.npz --rerankers noop,heuristic
```

注：cross-encoder 一列需要 `sentence-transformers` 与 `BAAI/bge-reranker-base`（约 1.1 GB）；离线 fixture 只覆盖稠密分支向量，不覆盖重排模型。
