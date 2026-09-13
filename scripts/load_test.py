#!/usr/bin/env python3
"""压测：对运行中的服务并发发问，统计 QPS 与延迟分位。

用法：
    python -m kb_agent serve --port 8000        # 另一个终端
    python scripts/load_test.py --url http://127.0.0.1:8000 --concurrency 8 --requests 200

默认从评测集里取问题，保证压测流量分布贴近真实查询。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_questions(path: Path, limit: int) -> list[str]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(json.loads(line)["question"])
    return questions[:limit]


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, int(round(ratio * (len(ordered) - 1))))
    return ordered[position]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="kb-agent 压测")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--dataset", default=str(ROOT / "eval" / "dataset.jsonl"))
    args = parser.parse_args()

    import httpx

    questions = load_questions(Path(args.dataset), args.requests)
    if not questions:
        print("评测集为空", file=sys.stderr)
        return 2

    latencies: list[float] = []
    failures = 0

    def call(index: int) -> float | None:
        question = questions[index % len(questions)]
        started = time.perf_counter()
        try:
            response = httpx.post(f"{args.url}/ask", json={"question": question, "use_cache": False}, timeout=60.0)
            response.raise_for_status()
        except Exception:
            return None
        return (time.perf_counter() - started) * 1000

    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for result in pool.map(call, range(args.requests)):
            if result is None:
                failures += 1
            else:
                latencies.append(result)
    elapsed = time.perf_counter() - started_at

    print(f"并发 {args.concurrency}｜请求 {args.requests}｜失败 {failures}")
    print(f"总耗时 {elapsed:.2f}s｜QPS {len(latencies) / elapsed:.1f}")
    if latencies:
        print(
            "延迟 ms："
            f"平均 {statistics.fmean(latencies):.1f}｜"
            f"P50 {percentile(latencies, 0.50):.1f}｜"
            f"P95 {percentile(latencies, 0.95):.1f}｜"
            f"P99 {percentile(latencies, 0.99):.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
