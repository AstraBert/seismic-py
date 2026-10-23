import heapq
import json
import random
from typing import TypedDict, cast

TokenVector = dict[str, float]
IdVector = dict[int, float]
Vector = TokenVector | IdVector


class RawEntry(TypedDict):
    id: int
    content: str | None
    vector: TokenVector


class Entry(TypedDict):
    content: str | None
    vector: TokenVector


def load_data(
    file: str, token_to_id_map: dict[str, int] | None = None
) -> tuple[dict[int, Entry], dict[str, int]]:
    token_to_id_map = token_to_id_map.copy() if token_to_id_map is not None else {}
    entries: dict[int, Entry] = {}
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            entry: RawEntry = json.loads(line)
            id_ = cast(int, entry.pop("id"))  # ty: ignore[invalid-argument-type]
            entries[id_] = cast(Entry, entry)
            for k in entry["vector"]:
                if k not in token_to_id_map:
                    token_to_id_map[k] = len(token_to_id_map) + 1
    return entries, token_to_id_map


def dot(a: Vector, b: Vector) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum([v * b[c] for (c, v) in a.items() if c in b])  # type: ignore


def brute_force_on_ids(
    documents: dict[int, Entry], query: Vector, k: int = 10
) -> list[int]:
    scored = [(dot(query, documents[i]["vector"]), i) for i in documents]
    return [d for s, d in heapq.nlargest(k, (p for p in scored if p[0] > 0))]


def recall(exact: list[int], retrieved: list[int]) -> float | None:
    if not exact:
        # nothing to find. A non-empty result here means there is
        # an error
        if len(retrieved) > 0:
            raise RuntimeError(
                "documents have been retrieved, but brute force ground truth is empty"
            )
        return None
    return len(set(exact) & set(retrieved[: len(exact)])) / len(exact)


def make_fake_data(
    n_docs: int = 3000,
    n_queries: int = 100,
    dim: int = 300,
    doc_nnz: tuple[int, int] = (8, 25),
    query_nnz: tuple[int, int] = (5, 12),
    zipf_s: float = 1.1,
    n_topics: int = 20,
    topic_size: int = 40,
    topic_mix: float = 0.6,
    seed: int = 0,
    as_raw: bool = False,
) -> tuple[dict[int, Entry] | list[RawEntry], dict[int, Entry] | list[RawEntry]]:
    rng = random.Random(seed)
    components = list(range(dim))
    weights = [1 / (rank + 1) ** zipf_s for rank in components]

    # each topic favors its own random subset of components (drawn with the Zipf law)
    topics = []
    for _ in range(n_topics):
        subset: set[int] = set()
        while len(subset) < min(topic_size, dim):
            subset.add(rng.choices(components, weights)[0])
        topics.append(sorted(subset))

    def make_vector(nnz: int) -> Vector:
        topic = rng.choice(topics)
        vector: Vector = {}
        while len(vector) < nnz:  # distinct components, so loop until we have nnz
            if rng.random() < topic_mix:
                c = rng.choice(topic)
            else:
                c = rng.choices(components, weights)[0]
            if f"t{c}" not in vector:
                vector[f"t{c}"] = round(rng.expovariate(1.5) + 0.01, 4)  # pyright: ignore[reportArgumentType]
        return vector

    docs = [make_vector(rng.randint(*doc_nnz)) for _ in range(n_docs)]
    queries = [make_vector(rng.randint(*query_nnz)) for _ in range(n_queries)]
    if as_raw:
        raw_doc_entries: list[RawEntry] = [
            {"id": i, "vector": cast(TokenVector, v), "content": None}
            for (i, v) in enumerate(docs)
        ]
        raw_query_entries: list[RawEntry] = [
            {"id": i, "vector": cast(TokenVector, v), "content": None}
            for (i, v) in enumerate(queries)
        ]
        return raw_doc_entries, raw_query_entries
    doc_entries = {
        i: Entry(content=None, vector=cast(TokenVector, v))
        for (i, v) in enumerate(docs)
    }
    query_entries = {
        i: Entry(content=None, vector=cast(TokenVector, v))
        for (i, v) in enumerate(queries)
    }
    return doc_entries, query_entries
