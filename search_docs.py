"""Самостоятельное сравнение поиска по примеру 02_three_searches.py из практики."""
import argparse
import json
import os
from pathlib import Path

from context_support import Docs, context_message
from search_support import SEARCH_MODES, Embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--knowledge", type=Path)
    parser.add_argument("--mode", choices=(*SEARCH_MODES, "all"), default="bm25")
    parser.add_argument("--questions", type=Path, help="JSON из практики: question, expect, kind (необязательно)")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-model")
    args = parser.parse_args()
    if not args.query and not args.questions:
        parser.error("Укажи запрос или --questions")
    embeddings = None
    if args.mode != "bm25":
        if not args.embedding_base_url or not args.embedding_model:
            parser.error("Для embedding/hybrid/all укажи сервер и модель эмбеддингов; автоматической подмены на BM25 нет")
        from openai import OpenAI
        embeddings = Embeddings(OpenAI(base_url=args.embedding_base_url,
                                       api_key=os.environ.get("EMBEDDING_API_KEY", "local"),
                                       max_retries=0), args.embedding_model)
    modes = SEARCH_MODES if args.mode == "all" else (args.mode,)
    for mode in modes:
        docs = Docs(Path(args.workspace).resolve(), mode, embeddings, args.knowledge)
        if args.query:
            found = docs.search(args.query, k=5)
            print(f"\n{mode}: " + json.dumps([{k: c[k] for k in ("id", "source", "score")} for c in found], ensure_ascii=False))
            print(context_message(found) or "Совпадений нет")
        if args.questions:
            questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
            if not isinstance(questions, list) or not all(
                isinstance(q, dict) and isinstance(q.get("question"), str)
                and isinstance(q.get("expect"), str) and q["expect"] for q in questions
            ):
                parser.error("Ожидается список вопросов с непустым expect")
            hits = 0
            by_kind = {}
            for q in questions:
                found = docs.search(q["question"], k=5)
                hit = any(q["expect"] in c["text"] for c in found)
                hits += hit
                group = by_kind.setdefault(str(q.get("kind", "other")), [0, 0])
                group[0] += hit
                group[1] += 1
            print(json.dumps(dict(mode=mode, metric="recall@5 (expect substring)",
                                  hits=hits, questions=len(questions), by_kind=by_kind), ensure_ascii=False))
    if embeddings:
        print(json.dumps(dict(embedding_requests=embeddings.requests,
                              reported_tokens=embeddings.reported_tokens,
                              usage_complete=embeddings.requests == embeddings.usage_responses)))


if __name__ == "__main__":
    main()
