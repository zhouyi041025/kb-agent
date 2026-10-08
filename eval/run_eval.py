"""评测：检索质量 + 拒答能力 + 引用准确率，输出消融对比表。

默认全程离线（哈希向量 + 抽取式 stub LLM），所以任何人 clone 下来都能复现同样的数字。
切换成真实模型只需要设置 KB_EMBEDDING_PROVIDER / KB_LLM_PROVIDER。

用法：
    python eval/run_eval.py
    python eval/run_eval.py --out eval/report.md --sweep
    python eval/run_eval.py --sweep --sensitivity --strategies        # 全量消融
    python eval/run_eval.py --split test                              # 只跑 test 划分
    python eval/run_eval.py --embedding-cache eval/fixtures/embedding-3.npz  # 离线复现语义向量评测
    python eval/run_eval.py --embedding-provider local --rerankers noop,heuristic,cross-encoder
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kb_agent.config import AppConfig, EmbeddingConfig  # noqa: E402
from kb_agent.embedder import (  # noqa: E402
    DEFAULT_LOCAL_MODEL,
    CachedEmbedder,
    HashingEmbedder,
    build_embedder,
)
from kb_agent.index import KnowledgeIndex  # noqa: E402
from kb_agent.ingest import build_chunks, load_documents  # noqa: E402
from kb_agent.llm import StubLLM  # noqa: E402
from kb_agent.rerank import CrossEncoderReranker, HeuristicReranker, NoopReranker  # noqa: E402
from kb_agent.service import REFUSAL_TEXT, KbService  # noqa: E402

CONFIGS = [
    ("纯向量检索", "vector", False),
    ("纯 BM25", "bm25", False),
    ("纯向量 + 重排", "vector", True),
    ("纯 BM25 + 重排", "bm25", True),
    ("混合检索 RRF", "hybrid", False),
    ("混合检索 RRF + 重排", "hybrid", True),
]


def load_dataset(path: Path) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items:
        raise SystemExit(f"评测集为空：{path}")
    return items


def build_index(
    knowledge_dir: Path, dim: int = 4096, fusion_weights=None, embedder=None, strategy: str = "recursive"
) -> KnowledgeIndex:
    documents = load_documents(knowledge_dir)
    chunks = build_chunks(documents, strategy=strategy)
    index = KnowledgeIndex.build(chunks, embedder or HashingEmbedder(dim=dim))
    if fusion_weights:
        index.fusion_weights = list(fusion_weights)
    return index


def make_service(
    index: KnowledgeIndex,
    mode: str,
    use_rerank: bool,
    top_k: int,
    min_confidence: float,
    reranker=None,
) -> KbService:
    config = AppConfig()
    config.retrieval.mode = mode
    config.retrieval.top_k = top_k
    config.retrieval.use_rerank = use_rerank
    config.retrieval.min_confidence = min_confidence
    return KbService(config, index=index, llm=StubLLM(), reranker=reranker)


def build_reranker(name: str, model: str):
    """按名字构造重排器；默认启发式，可切换到 noop（对照）与 cross-encoder。"""
    if name == "noop":
        return NoopReranker()
    if name == "cross-encoder":
        return CrossEncoderReranker(model)
    return HeuristicReranker()


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, int(round(ratio * (len(ordered) - 1))))
    return round(ordered[position], 2)


def evaluate(
    index: KnowledgeIndex,
    dataset: list[dict],
    mode: str,
    use_rerank: bool,
    top_k: int,
    min_confidence: float,
    reranker=None,
    reranker_name: str = "heuristic",
) -> dict:
    service = make_service(index, mode, use_rerank, top_k, min_confidence, reranker=reranker)
    rows: list[dict] = []
    for item in dataset:
        answer = service.answer(item["question"], mode=mode, top_k=top_k, use_rerank=use_rerank, use_cache=False)
        retrieved = [context.chunk.doc_id for context in answer.contexts]
        gold = set(item["gold_docs"])
        first_relevant = next((rank for rank, doc in enumerate(retrieved, start=1) if doc in gold), None)
        cited_docs = {
            context.chunk.doc_id for context in answer.contexts if context.chunk.chunk_id in set(answer.citations)
        }
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answerable": item["answerable"],
                "refused": answer.answer.strip() == REFUSAL_TEXT,
                "retrieved": retrieved,
                "gold": sorted(gold),
                "hit@1": bool(first_relevant == 1),
                "recall@k": (len(set(retrieved) & gold) / len(gold)) if gold else None,
                "reciprocal_rank": (1 / first_relevant) if first_relevant else 0.0,
                "citation_precision": (len(cited_docs & gold) / len(cited_docs)) if cited_docs else None,
                "latency_ms": answer.latency_ms,
            }
        )
    return summarize(rows, mode, use_rerank, top_k, min_confidence, reranker_name)


def summarize(
    rows: list[dict],
    mode: str,
    use_rerank: bool,
    top_k: int,
    min_confidence: float,
    reranker_name: str = "heuristic",
) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    refusal_correct = [row for row in unanswerable if row["refused"]]
    missed = [row for row in answerable if row["refused"]]
    citation_scores = [row["citation_precision"] for row in answerable if row["citation_precision"] is not None and not row["refused"]]
    latencies = [row["latency_ms"] for row in rows if row.get("latency_ms") is not None]
    return {
        "mode": mode,
        "use_rerank": use_rerank,
        "reranker": reranker_name,
        "top_k": top_k,
        "min_confidence": min_confidence,
        "hit@1": statistics.fmean(row["hit@1"] for row in answerable) if answerable else 0.0,
        f"recall@{top_k}": statistics.fmean(row["recall@k"] for row in answerable) if answerable else 0.0,
        "mrr": statistics.fmean(row["reciprocal_rank"] for row in answerable) if answerable else 0.0,
        "refusal_accuracy": len(refusal_correct) / len(unanswerable) if unanswerable else 0.0,
        "false_answer_rate": 1 - (len(refusal_correct) / len(unanswerable)) if unanswerable else 0.0,
        "miss_rate": len(missed) / len(answerable) if answerable else 0.0,
        "citation_precision": statistics.fmean(citation_scores) if citation_scores else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "rows": rows,
    }


def format_table(results: list[dict], top_k: int) -> str:
    header = (
        f"| 配置 | Recall@1 | Recall@{top_k} | MRR | 引用准确率 | 拒答正确率 | 误答率 | 漏答率 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [header]
    for result in results:
        lines.append(
            "| {name} | {hit:.3f} | {recall:.3f} | {mrr:.3f} | {citation:.3f} | {refusal:.3f} | {false_rate:.3f} | {miss:.3f} |".format(
                name=result["label"],
                hit=result["hit@1"],
                recall=result[f"recall@{top_k}"],
                mrr=result["mrr"],
                citation=result["citation_precision"],
                refusal=result["refusal_accuracy"],
                false_rate=result["false_answer_rate"],
                miss=result["miss_rate"],
            )
        )
    return "\n".join(lines)


def format_sweep(sweep: list[dict]) -> str:
    lines = [
        "| 置信度阈值 | 拒答正确率 | 误答率 | 漏答率 |",
        "| --- | --- | --- | --- |",
    ]
    for item in sweep:
        lines.append(
            "| {threshold:.2f} | {refusal:.3f} | {false_rate:.3f} | {miss:.3f} |".format(
                threshold=item["min_confidence"],
                refusal=item["refusal_accuracy"],
                false_rate=item["false_answer_rate"],
                miss=item["miss_rate"],
            )
        )
    return "\n".join(lines)


def format_sensitivity(rows: list[dict], top_k: int) -> str:
    lines = [
        "| 向量维度 | 纯向量 R@1 | 纯向量+重排 R@1 | 纯 BM25+重排 R@1 | 混合+重排 R@1 | 混合+重排 MRR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {dim} | {vector:.3f} | {vector_rr:.3f} | {bm25_rr:.3f} | {hybrid_rr:.3f} | {hybrid_mrr:.3f} |".format(
                dim=row["dim"],
                vector=row["vector"],
                vector_rr=row["vector_rerank"],
                bm25_rr=row["bm25_rerank"],
                hybrid_rr=row["hybrid_rerank"],
                hybrid_mrr=row["hybrid_mrr"],
            )
        )
    return "\n".join(lines)


def format_fusion(rows: list[dict]) -> str:
    lines = [
        "| 融合权重 [BM25, 向量] | 混合 R@1 | 混合 MRR | 混合+重排 R@1 | 混合+重排 MRR |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {weights} | {hybrid:.3f} | {hybrid_mrr:.3f} | {hybrid_rr:.3f} | {hybrid_rr_mrr:.3f} |".format(
                weights=row["weights"],
                hybrid=row["hybrid"],
                hybrid_mrr=row["hybrid_mrr"],
                hybrid_rr=row["hybrid_rerank"],
                hybrid_rr_mrr=row["hybrid_rerank_mrr"],
            )
        )
    return "\n".join(lines)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="kb-agent 评测")
    parser.add_argument("--dataset", default=str(ROOT / "eval" / "dataset.jsonl"))
    parser.add_argument("--knowledge-dir", default=str(ROOT / "data" / "knowledge"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--embedding-dim", type=int, default=4096)
    parser.add_argument(
        "--embedding-provider",
        choices=["hashing", "openai", "local"],
        default="hashing",
        help=(
            "hashing=离线确定性（默认，报告可复现）；"
            "local=本地语义向量（sentence-transformers，无需 Key）；"
            "openai=远端语义向量（读 .env 里的 Key，结果随模型版本漂移）"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="语义向量的模型名：local 默认 BAAI/bge-small-zh-v1.5；openai 例如 embedding-3（必填）",
    )
    parser.add_argument(
        "--reranker",
        choices=["noop", "heuristic", "cross-encoder"],
        default="heuristic",
        help="重排器：noop=不重排（对照）/ heuristic=启发式（默认）/ cross-encoder=交叉编码器（需 sentence-transformers）",
    )
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base", help="cross-encoder 使用的模型名")
    parser.add_argument(
        "--rerankers",
        default="",
        help="额外做重排器网格对比，逗号分隔（如 noop,heuristic,cross-encoder），结果写入报告新章节",
    )
    parser.add_argument("--sweep", action="store_true", help="额外跑一次拒答阈值扫参")
    parser.add_argument("--sensitivity", action="store_true", help="额外跑向量维度与融合权重消融")
    parser.add_argument("--strategies", action="store_true", help="额外跑切分策略消融（recursive / fixed / parent_child）")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all", help="只评测某个划分")
    parser.add_argument(
        "--embedding-cache",
        default="",
        help="离线向量 fixture（npz，由 scripts/make_embedding_fixture.py 生成）：用它替代远端调用，让语义向量评测可复现",
    )
    parser.add_argument("--dump", default="", help="把首选配置的逐条结果写成 JSONL，便于定位失败样本")
    parser.add_argument("--out", default=str(ROOT / "eval" / "report.md"))
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    if args.split != "all":
        dataset = [item for item in dataset if item.get("split", "dev") == args.split]
        if not dataset:
            raise SystemExit(f"划分 {args.split} 里没有任何用例")
    knowledge_dir = Path(args.knowledge_dir)
    embedder = None
    if args.embedding_cache:
        embedder = CachedEmbedder(args.embedding_cache)
    elif args.embedding_provider == "openai":
        if not args.embedding_model:
            raise SystemExit("--embedding-provider openai 需要同时给 --embedding-model（例如 embedding-3）")
        config = AppConfig.from_env()
        config.embedding.provider = "openai"
        config.embedding.model = args.embedding_model
        embedder = build_embedder(
            config.embedding, base_url=config.llm.base_url, api_key=config.llm.api_key
        )
    elif args.embedding_provider == "local":
        embedder = build_embedder(
            EmbeddingConfig(provider="local", model=args.embedding_model or DEFAULT_LOCAL_MODEL)
        )
    index = build_index(knowledge_dir, dim=args.embedding_dim, embedder=embedder)
    total_docs = len({chunk.doc_id for chunk in index.chunks})
    negatives = sum(1 for item in dataset if not item["answerable"])
    reranker = build_reranker(args.reranker, args.reranker_model)
    reranker_label = args.reranker + (f"（{args.reranker_model}）" if args.reranker == "cross-encoder" else "")
    if embedder is None:
        embedder_label = f"{index.embedder.name}（离线确定性，哈希词法向量）"
        reproducibility_note = "所有指标均可在无网络、无 API Key 的环境下复现。"
    elif str(index.embedder.name).startswith("local-"):
        embedder_label = f"{index.embedder.name}（本地语义向量）"
        reproducibility_note = (
            "本次使用本地语义向量（sentence-transformers）。同一模型版本下结果稳定；"
            "若仓库内提供对应 `eval/fixtures/*.npz`，用 `--embedding-cache` 可完全离线复现。"
        )
    else:
        embedder_label = f"{index.embedder.name}（远端语义向量，数字随模型版本漂移）"
        reproducibility_note = (
            "本次的稠密分支用了远端语义向量，**需要 API Key**；"
            "离线方案见 `--embedding-provider local` 与其 fixture。"
        )
    print(
        f"知识库：{total_docs} 篇文档 / {len(index.chunks)} 个片段；"
        f"评测集：{len(dataset)} 条查询（划分 {args.split}，负样本 {negatives} 条）；"
        f"向量模型：{index.embedder.name}；重排器：{reranker_label}"
    )

    results = []
    for label, mode, use_rerank in CONFIGS:
        result = evaluate(
            index,
            dataset,
            mode,
            use_rerank,
            args.top_k,
            args.min_confidence,
            reranker=reranker,
            reranker_name=args.reranker,
        )
        result["label"] = label
        results.append(result)
        print(f"  ✓ {label}")

    sweep: list[dict] = []
    if args.sweep:
        for threshold in (0.0, 0.25, 0.35, 0.45, 0.55, 0.65):
            item = evaluate(index, dataset, "hybrid", True, args.top_k, threshold)
            sweep.append(item)
        print("  ✓ 拒答阈值扫参")

    sensitivity: list[dict] = []
    fusion_rows: list[dict] = []
    if args.sensitivity:
        # 维度消融只对哈希向量有意义：真实语义向量的维度由模型固定，改不了
        dims = (512, 2048, 4096) if embedder is None else ()
        for dim in dims:
            candidate = build_index(knowledge_dir, dim=dim)
            sensitivity.append(
                {
                    "dim": dim,
                    "vector": evaluate(candidate, dataset, "vector", False, args.top_k, args.min_confidence)["hit@1"],
                    "vector_rerank": evaluate(candidate, dataset, "vector", True, args.top_k, args.min_confidence)["hit@1"],
                    "bm25_rerank": evaluate(candidate, dataset, "bm25", True, args.top_k, args.min_confidence)["hit@1"],
                    "hybrid_rerank": evaluate(candidate, dataset, "hybrid", True, args.top_k, args.min_confidence)["hit@1"],
                    "hybrid_mrr": evaluate(candidate, dataset, "hybrid", True, args.top_k, args.min_confidence)["mrr"],
                }
            )
        for weights in ([1.0, 1.0], [1.0, 2.0], [2.0, 1.0]):
            candidate = build_index(
                knowledge_dir, dim=args.embedding_dim, fusion_weights=weights, embedder=embedder
            )
            plain = evaluate(candidate, dataset, "hybrid", False, args.top_k, args.min_confidence)
            reranked = evaluate(candidate, dataset, "hybrid", True, args.top_k, args.min_confidence)
            fusion_rows.append(
                {
                    "weights": weights,
                    "hybrid": plain["hit@1"],
                    "hybrid_mrr": plain["mrr"],
                    "hybrid_rerank": reranked["hit@1"],
                    "hybrid_rerank_mrr": reranked["mrr"],
                }
            )
        print("  ✓ 向量维度与融合权重消融")

    strategy_rows: list[dict] = []
    if args.strategies:
        for strategy in ("recursive", "fixed", "parent_child"):
            candidate = build_index(knowledge_dir, dim=args.embedding_dim, embedder=embedder, strategy=strategy)
            hybrid = evaluate(candidate, dataset, "hybrid", True, args.top_k, args.min_confidence)
            bm25 = evaluate(candidate, dataset, "bm25", True, args.top_k, args.min_confidence)
            vector = evaluate(candidate, dataset, "vector", True, args.top_k, args.min_confidence)
            strategy_rows.append(
                {
                    "strategy": strategy,
                    "chunks": len(candidate.chunks),
                    "hybrid": hybrid["hit@1"],
                    "hybrid_mrr": hybrid["mrr"],
                    "bm25": bm25["hit@1"],
                    "vector": vector["hit@1"],
                }
            )
        print("  ✓ 切分策略消融")

    reranker_rows: list[dict] = []
    if args.rerankers:
        for name in [part.strip() for part in args.rerankers.split(",") if part.strip()]:
            candidate = evaluate(
                index,
                dataset,
                "hybrid",
                True,
                args.top_k,
                args.min_confidence,
                reranker=build_reranker(name, args.reranker_model),
                reranker_name=name,
            )
            reranker_rows.append(
                {
                    "reranker": name,
                    "hit@1": candidate["hit@1"],
                    "mrr": candidate["mrr"],
                    "citation": candidate["citation_precision"],
                    "p50": candidate["p50_latency_ms"],
                    "p95": candidate["p95_latency_ms"],
                }
            )
        print("  ✓ 重排器网格对比")

    if args.dump and results:
        with open(args.dump, "w", encoding="utf-8") as handle:
            for row in results[0]["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"逐条结果已写入 {args.dump}")

    table = format_table(results, args.top_k)
    print("\n" + table + "\n")

    lines = [
        "# 评测报告",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- 知识库：{total_docs} 篇文档，切分为 {len(index.chunks)} 个片段（recursive 策略）",
        f"- 评测集：{len(dataset)} 条查询（划分：{args.split}），其中可回答 {sum(1 for i in dataset if i['answerable'])} 条，"
        f"不可回答 {sum(1 for i in dataset if not i['answerable'])} 条",
        f"- 向量模型：{embedder_label}｜生成模型：stub-extractive（离线抽取式）",
        f"- 重排器：{reranker_label}",
        "- 评测命令：`python eval/run_eval.py"
        + f" --top-k {args.top_k} --min-confidence {args.min_confidence}"
        + (
            f" --embedding-provider {args.embedding_provider} --embedding-model {args.embedding_model}"
            if embedder is not None
            else ""
        )
        + (" --sweep" if args.sweep else "")
        + (" --sensitivity" if args.sensitivity else "")
        + (" --strategies" if args.strategies else "")
        + (f" --rerankers {args.rerankers}" if args.rerankers else "")
        + "`",
        "",
        reproducibility_note,
        "",
        f"## 消融对比（top_k={args.top_k}, 拒答阈值={args.min_confidence}）",
        "",
        table,
        "",
        "指标说明：",
        "",
        "- Recall@1 / MRR：排序质量，MRR 奖励把正确文档排得更靠前",
        "- 引用准确率：答案标注的引用里，落在标准答案文档上的比例",
        "- 拒答正确率：知识库确实没有答案时正确拒答的比例",
        "- 误答率 = 1 - 拒答正确率，是最危险的错误类型（没有依据却给出答案）",
        "- 漏答率：知识库有答案却拒答的比例，与误答率是一对取舍",
    ]
    if sweep:
        lines += ["", "## 拒答阈值扫参（混合检索 RRF + 重排）", "", format_sweep(sweep), "",
                  "阈值越高越保守：误答率下降，漏答率上升。选哪个点取决于业务对错误答案的容忍度。"]
    if sensitivity:
        lines += [
            "",
            "## 向量维度消融",
            "",
            format_sensitivity(sensitivity, args.top_k),
            "",
            "哈希向量的维度决定签名哈希的碰撞率。512 维时碰撞严重，稠密分支质量被拖累，"
            "混合检索反而不如纯 BM25；维度提升后稠密分支可用，结论随之改变。"
            "这说明**检索组件的效果不是固定属性，取决于组件的实际质量**，"
            "任何「加一路召回就一定更好」的假设都要用评测推翻或确认。",
        ]
    if fusion_rows:
        lines += [
            "",
            "## 融合权重消融",
            "",
            format_fusion(fusion_rows),
            "",
            (
                "在本语料上加权融合并未带来稳定增益：两路召回都是词法信号、高度相关，"
                "融合只是在重新分配同一批候选的名次。要真正拉开差距，需要让两路互补——"
                "也就是把稠密分支换成真正的语义向量（`--embedding-provider openai`）。"
                if embedder is None
                else "换成语义向量后两路分支不再高度相关，但加权仍然挤不出增益："
                "等权重已接近最优，把权重压向 BM25 一侧反而下降。"
                "融合权重能起多大作用，取决于两路分支的质量差距。"
            ),
            "",
            "> 方法学说明：评测集带 `split` 字段（q001–q075 为 dev，q076–q150 为 test）。"
            "本报告默认跑全量集以便与历史数字对比；选超参请用 `--split dev`，对外报告用 `--split test`。"
            "当前规模下敏感性分析只用于方向性判断，不适合精调超参。",
        ]

    if strategy_rows:
        lines += [
            "",
            "## 切分策略消融",
            "",
            "| 切分策略 | 片段数 | 纯 BM25+重排 R@1 | 纯向量+重排 R@1 | 混合+重排 R@1 | 混合+重排 MRR |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in strategy_rows:
            lines.append(
                "| {strategy} | {chunks} | {bm25:.3f} | {vector:.3f} | {hybrid:.3f} | {mrr:.3f} |".format(
                    strategy=row["strategy"],
                    chunks=row["chunks"],
                    bm25=row["bm25"],
                    vector=row["vector"],
                    hybrid=row["hybrid"],
                    mrr=row["hybrid_mrr"],
                )
            )
        lines += [
            "",
            "注意：本表用的是文档级召回指标，长片段更容易恰好覆盖 gold 文档，",
            "因此 fixed 略好并不等于它更适合生成（片段过长会稀释上下文）。",
            "切分的真实取舍要结合片段级指标与生成质量一起看；这里先把它当作一个待解释的观察。",
        ]

    if reranker_rows:
        lines += [
            "",
            f"## 重排器对比（混合检索，top_k={args.top_k}）",
            "",
            "| 重排器 | R@1 | MRR | 引用准确率 | P50 延迟 | P95 延迟 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in reranker_rows:
            lines.append(
                "| {name} | {hit:.3f} | {mrr:.3f} | {citation:.3f} | {p50:.1f} ms | {p95:.1f} ms |".format(
                    name=row["reranker"],
                    hit=row["hit@1"],
                    mrr=row["mrr"],
                    citation=row["citation"],
                    p50=row["p50"],
                    p95=row["p95"],
                )
            )
        lines += [
            "",
            "延迟是端到端单题耗时的分位（含 stub 生成）。cross-encoder 的精度增益要放在这条延迟曲线上权衡；",
            "重排的上界仍由召回决定——候选里没有正确文档时，换多强的重排器都没用。",
        ]

    report = "\n".join(lines) + "\n"
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
