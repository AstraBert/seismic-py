import heapq
from collections import defaultdict
from typing import TypedDict, cast

from .data import Entry, Vector, brute_force_on_ids, dot, load_data

DEFAULT_D = 16
DEFAULT_L = 16
DEFAULT_T = 2


class GripIndex(TypedDict):
    wmaxes: dict[str, float]
    h_lists: dict[str, list[int]]
    graph: Graph
    full_vecs: dict[int, Vector]
    sketch_vecs: dict[int, Vector]
    default_entry_point: int


Graph = dict[int, list[int]]


def exact_top_10(documents: dict[int, Entry], queries: list[Vector]) -> list[list[int]]:
    ground_truths = []
    for q in queries:
        ground_truths.append(brute_force_on_ids(documents, q, k=10))
    return ground_truths


def build_graph(
    documents: dict[int, Vector], d: int = DEFAULT_D
) -> tuple[Graph, dict[str, dict[int, float]]]:
    posting_list: dict[str, dict[int, float]] = {}
    for item in documents.items():
        for k, v in item[1].items():
            posting_list.setdefault(cast(str, k), {})[item[0]] = v
    graph: Graph = {}
    for doc_id, entry in documents.items():
        acc: defaultdict[int, float] = defaultdict(float)
        for term, weight in entry.items():
            for other_doc_id, other_weight in posting_list[cast(str, term)].items():
                if other_doc_id != doc_id:
                    acc[other_doc_id] += weight * other_weight
        neighbours = sorted(acc.items(), key=lambda x: x[1], reverse=True)[:d]
        graph[doc_id] = [n[0] for n in neighbours]

    return graph, posting_list


def seed_list(
    posting: dict[str, dict[int, float]], l: int
) -> tuple[dict[str, float], dict[str, list[int]]]:
    wmaxs: dict[str, float] = {}
    h_lists: dict[str, list[int]] = {}

    for p, v in posting.items():
        max_w = 0
        top_docs: list[tuple[float, int]] = []
        for i, w in v.items():
            max_w = max(w, max_w)
            if len(top_docs) < l:
                heapq.heappush(top_docs, (w, i))
            elif w > top_docs[0][0]:
                heapq.heapreplace(top_docs, (w, i))
        wmaxs.update({p: max_w})
        top_docs.sort(key=lambda x: x[0], reverse=True)
        h_lists.update({p: [t[1] for t in top_docs]})

    wmaxs = {k: v for k, v in sorted(wmaxs.items(), key=lambda x: x[1], reverse=True)}
    return wmaxs, h_lists


def entry_points(
    query: Vector,
    seeds: dict[str, list[int]],
    wmaxs: dict[str, float],
    default_entry_point: int,
    t: int,
) -> set[int]:
    scored_terms: dict[str, float] = {}
    for q, v in query.items():
        if q in wmaxs:
            scored_terms[q] = v * wmaxs[q]
    if not scored_terms:
        return {default_entry_point}
    scored_t = heapq.nlargest(t, scored_terms.items(), key=lambda x: x[1])
    scored_t.sort(key=lambda x: x[1], reverse=True)
    seeded_list = set()
    for term, _ in scored_t:
        if term in seeds:
            for k in seeds[term]:
                seeded_list.add(k)

    return seeded_list


def make_sketch(vec: Vector, r: int) -> Vector:
    heap: list[tuple[float, str]] = []
    for k, v in vec.items():
        if len(heap) < r:
            heapq.heappush(heap, (v, cast(str, k)))
        elif len(heap) >= r and v > heap[0][0]:
            heapq.heapreplace(heap, (v, cast(str, k)))
    return {term: value for value, term in heap}


def beam_search(
    query: Vector,
    graph: Graph,
    routing_vecs: dict[int, Vector],
    entries: set[int],
    ef: int,
) -> list[int]:
    visited: set[int] = set(entries)
    candidates: list[tuple[float, int]] = []
    beam: list[tuple[float, int]] = []
    for entry in entries:
        score = dot(query, routing_vecs[entry])
        heapq.heappush(candidates, (-score, entry))
        if len(beam) < ef:
            heapq.heappush(beam, (score, entry))
        elif len(beam) >= ef and score > beam[0][0]:
            heapq.heapreplace(beam, (score, entry))
    while len(candidates) > 0:
        c = heapq.heappop(candidates)
        score = -c[0]
        if len(beam) == ef and score <= beam[0][0]:
            break
        for nb in graph[c[1]]:
            if nb not in visited:
                visited.add(nb)
                nb_score = dot(query, routing_vecs[nb])
                if len(beam) < ef:
                    heapq.heappush(beam, (nb_score, nb))
                    heapq.heappush(candidates, (-nb_score, nb))
                elif nb_score > beam[0][0]:
                    heapq.heapreplace(beam, (nb_score, nb))
                    heapq.heappush(candidates, (-nb_score, nb))
    return [i for _, i in beam]


def build_index(
    documents_path: str,
    graph_d: int = DEFAULT_D,
    seed_l: int = DEFAULT_L,
    sketch_r: int = 8,
) -> GripIndex:
    entries, _ = load_data(documents_path)
    documents = {i: cast(Vector, entry["vector"]) for i, entry in entries.items()}
    default_entry_point = max(
        documents.keys(), key=lambda i: sum(documents[i].values())
    )
    graph, posting_list = build_graph(
        documents=cast(dict[int, Vector], documents), d=graph_d
    )
    wmaxes, h_lists = seed_list(posting_list, seed_l)
    routing_vecs = {i: make_sketch(vec, sketch_r) for i, vec in documents.items()}

    return GripIndex(
        wmaxes=wmaxes,
        h_lists=h_lists,
        sketch_vecs=routing_vecs,
        graph=graph,
        default_entry_point=default_entry_point,
        full_vecs=documents,
    )


def grip_search(
    index: GripIndex,
    query: Vector,
    ef: int,
    use_seeds: bool,
    use_sketches: bool,
    k: int = 10,
    entrypoint_t: int = 2,
) -> list[int]:
    if use_seeds:
        entries = entry_points(
            query,
            seeds=index["h_lists"],
            wmaxs=index["wmaxes"],
            default_entry_point=index["default_entry_point"],
            t=entrypoint_t,
        )
    else:
        entries = {index["default_entry_point"]}

    if use_sketches:
        routing_vecs = index["sketch_vecs"]
    else:
        routing_vecs = index["full_vecs"]

    ids = beam_search(query, index["graph"], routing_vecs, entries, ef)
    scores: dict[int, float] = {}
    for id_ in ids:
        score = dot(query, index["full_vecs"][id_])
        scores[id_] = score

    rescored = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [r[0] for r in rescored][:k]
