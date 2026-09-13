#!/usr/bin/env python3
"""重建索引：python scripts/build_index.py [--knowledge-dir data/knowledge] [--index-dir .kb_index]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kb_agent.config import AppConfig  # noqa: E402
from kb_agent.service import KbService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="重建知识库索引")
    parser.add_argument("--knowledge-dir", default=None)
    parser.add_argument("--index-dir", default=None)
    args = parser.parse_args()

    config = AppConfig.from_env()
    if args.knowledge_dir:
        config.knowledge_dir = args.knowledge_dir
    if args.index_dir:
        config.index_dir = args.index_dir

    service = KbService(config)
    result = service.ingest()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"向量模型：{service.embedder.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
