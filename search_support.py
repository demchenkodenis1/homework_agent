"""Алгоритмы практики 1.3: BM25, cosine similarity и гибрид RRF.

BM25 работает без дополнительных пакетов. Эмбеддинги подключаются явно,
только для локальных экспериментов; ключ оценочного шлюза не используется.
"""
import math
import re
import time
from collections import Counter, defaultdict

SEARCH_MODES = ("bm25", "embedding", "hybrid")
STOP_WORDS = set("the and for with this that from into и в на с по у к о а но что как это для из".split())


def words(text):
    # Разбиваем snake_case и CamelCase, сохраняя номера и короткие термины API.
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [w for w in re.findall(r"[^\W_]+", text.lower()) if w not in STOP_WORDS]


class WordSearch:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.counts = [Counter(words(c["section"] + "\n" + c["text"])) for c in chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / max(1, len(chunks)) or 1
        df = Counter(w for counts in self.counts for w in counts)
        self.idf = {w: math.log(1 + (len(chunks) - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def search(self, query, k=5):
        scored = []
        for chunk, counts, length in zip(self.chunks, self.counts, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.average)
            score = sum(self.idf[w] * counts[w] * (self.k1 + 1) / (counts[w] + norm)
                        for w in set(words(query)) if counts[w])
            if score > 0:
                scored.append((chunk, score))
        return sorted(scored, key=lambda pair: (-pair[1], pair[0]["id"]))[:k]


def normalized(vector):
    if not vector or not all(type(v) in (int, float) and math.isfinite(v) for v in vector):
        raise ValueError("Некорректный вектор эмбеддинга")
    norm = math.sqrt(sum(v * v for v in vector))
    if not norm or not math.isfinite(norm):
        raise ValueError("Нулевой или некорректный вектор эмбеддинга")
    return [v / norm for v in vector]


class Embeddings:
    """Кэш в памяти одного запуска; никаких записей в /agent или workspace."""
    def __init__(self, client, model):
        self.client, self.model = client, model
        self.cache = {}
        self.dimension = None
        self.deadline = float("inf")
        self.requests = 0
        self.reported_tokens = 0
        self.usage_responses = 0

    def embed(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        for start in range(0, len(missing), 16):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Истёк лимит времени поиска")
            batch = missing[start:start + 16]
            self.requests += 1
            response = self.client.embeddings.create(
                model=self.model, input=batch, encoding_format="float", timeout=min(30, remaining))
            usage = getattr(getattr(response, "usage", None), "total_tokens", None)
            if type(usage) is int:
                self.reported_tokens += usage
                self.usage_responses += 1
            rows = sorted(response.data, key=lambda row: row.index)
            if [row.index for row in rows] != list(range(len(batch))):
                raise ValueError("Неполный ответ сервера эмбеддингов")
            vectors = [normalized(row.embedding) for row in rows]
            dimension = self.dimension or len(vectors[0])
            if any(len(v) != dimension for v in vectors):
                raise ValueError("Несовместимые размерности эмбеддингов")
            self.dimension = dimension
            self.cache.update(zip(batch, vectors))
        return [self.cache[t] for t in texts]


class MeaningSearch:
    def __init__(self, chunks, embeddings):
        self.chunks, self.embeddings = chunks, embeddings

    def search(self, query, k=5):
        if not self.chunks:
            return []
        vectors = self.embeddings.embed([c["section"] + "\n" + c["text"] for c in self.chunks])
        q = self.embeddings.embed([query])[0]
        scored = [(c, sum(a * b for a, b in zip(q, v))) for c, v in zip(self.chunks, vectors)]
        return sorted(scored, key=lambda pair: (-pair[1], pair[0]["id"]))[:k]


def hybrid(query, word_search, meaning_search, k=5):
    """RRF из практики: складываем ранги, а не несопоставимые оценки."""
    scores, chunks = defaultdict(float), {}
    for results in (word_search.search(query, 20), meaning_search.search(query, 20)):
        for rank, (chunk, _) in enumerate(results, 1):
            scores[chunk["id"]] += 1 / (60 + rank)
            chunks[chunk["id"]] = chunk
    best = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
    return [(chunks[cid], scores[cid]) for cid in best]
