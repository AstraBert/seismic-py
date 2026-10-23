import heapq
import random
from collections import defaultdict
from typing import Any, TypedDict

from .data import Entry, Vector, dot


class SearchResult(TypedDict):
    score: float
    doc_id: int


def global_threshold_pruning(
    docs: dict[int, Entry], dim: int, n_postings: int, max_fraction: float
) -> dict[str, list[int]]:
    values: list[float] = []
    components: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    for k, v in docs.items():
        for f, w in v["vector"].items():
            if w > 0:
                values.append(w)
                components[f].append((k, w))
    values.sort(reverse=True)
    total = min(dim * n_postings, len(values))
    lam = total - 1
    threshold = values[lam]
    cut = int(max_fraction * n_postings)
    for c in components:
        new_c = [v for v in components[c] if v[1] >= threshold]
        new_c.sort(key=lambda x: (-x[1], x[0]))
        components[c] = new_c[:cut]

    return {k: [el[0] for el in v] for k, v in components.items() if len(v) > 0}


def kmeans_blocks(
    doc_ids: list[int],
    docs: dict[int, Entry],
    centroid_fraction: float,
    min_cluster_size: int,
    rng: random.Random,
) -> list[list[int]]:
    n_centroids = max(1, int(centroid_fraction * len(doc_ids)))
    centroid_ids = rng.sample(doc_ids, k=n_centroids)
    blocks = {i: [i] for i in centroid_ids}
    for doc_id in doc_ids:
        if doc_id not in centroid_ids:
            dots: list[float] = []
            for centroid in centroid_ids:
                dots.append(dot(docs[doc_id]["vector"], docs[centroid]["vector"]))
            blocks[centroid_ids[dots.index(max(dots))]].append(doc_id)
    removed = []
    topop = []
    for block, value in blocks.items():
        if len(value) <= min_cluster_size:
            removed.extend(value)
            topop.append(block)
    if len(topop) == len(blocks):
        # all centroids removed because
        # too weak, return one full block
        return [doc_ids]
    new_blocks = {k: v for k, v in blocks.items() if k not in topop}
    new_centroids = list(new_blocks.keys())
    for rem in removed:
        dots = []
        for centroid in new_centroids:
            dots.append(dot(docs[rem]["vector"], docs[centroid]["vector"]))
        new_blocks[new_centroids[dots.index(max(dots))]].append(rem)

    return list(new_blocks.values())


def summarize(block: list[int], docs: dict[int, Entry], energy: float) -> Vector:
    components: defaultdict[str, list[float]] = defaultdict(list)
    for i in block:
        vec = docs[i]["vector"]
        for component, w in vec.items():
            components[component].append(w)
    maxed = {k: max(v) for k, v in components.items()}
    total = sum(list(maxed.values()))
    sort_maxed = {
        k: v for k, v in sorted(maxed.items(), key=lambda x: x[1], reverse=True)
    }
    tokeep = {}
    t = 0
    for m, v in sort_maxed.items():
        t += v
        tokeep[m] = v
        if t >= total * energy:
            break
    return tokeep


class Summaries:
    def __init__(
        self,
        n_summaries: int,
        compressed: defaultdict[str, list[tuple[int, int]]],
        quants: dict[int, float],
        mins: dict[int, float],
    ) -> None:
        self.n_summaries = n_summaries
        self.compressed = compressed
        self.quants = quants
        self.mins = mins

    @classmethod
    def from_summaries(cls, summaries: list[Vector]) -> Summaries:
        n_summaries = len(summaries)
        compressed: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        quants: dict[int, float] = {}
        mins: dict[int, float] = {}
        for i, summary in enumerate(summaries):
            vals = summary.values()
            minim = min(vals)
            maxim = max(vals)
            quant = (maxim - minim) / 255
            quants[i] = quant
            mins[i] = minim
            for k, v in summary.items():
                if quant != 0:
                    code = round((v - minim) / quant)
                else:
                    code = 0
                compressed[k].append((i, code))  # type: ignore
        return cls(n_summaries, compressed, quants, mins)

    def distances(self, query: Vector) -> list[float]:
        acc = [0.0] * self.n_summaries
        for c, qv in query.items():
            if c not in self.compressed:
                continue
            for summary_id, code in self.compressed[c]:  # type: ignore
                value = code * self.quants[summary_id] + self.mins[summary_id]
                acc[summary_id] += value * qv
        return acc

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_summaries": self.n_summaries,
            "compressed": self.compressed,
            "quants": self.quants,
            "mins": self.mins,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Summaries:
        n_summaries = d.get("n_summaries")
        if n_summaries is None:
            raise ValueError(
                "Dictionary does not contain required entry: 'n_summaries'"
            )
        compressed = d.get("compressed")
        if compressed is None:
            raise ValueError("Dictionary does not contain required entry: 'compressed'")
        quants = d.get("quants")
        if quants is None:
            raise ValueError("Dictionary does not contain required entry: 'quants'")
        mins = d.get("mins")
        if mins is None:
            raise ValueError("Dictionary does not contain required entry: 'mins'")
        return cls(n_summaries, compressed, quants, mins)


PostingLists = dict[str, tuple[list[int], list[int], Summaries]]


def build_posting_lists(
    docs: dict[int, Entry],
    dim: int,
    n_postings: int,
    max_fraction: float,
    centroid_fraction: float,
    min_cluster_size: int,
    rng: random.Random,
    energy: float,
) -> PostingLists:
    posting_lists: PostingLists = {}
    pruned = global_threshold_pruning(docs, dim, n_postings, max_fraction)
    for component, doc_ids in pruned.items():
        blocks = kmeans_blocks(doc_ids, docs, centroid_fraction, min_cluster_size, rng)
        docs_in_block_order = [b for block in blocks for b in block]
        block_offsets: list[int] = []
        current_offset = 0
        for b in blocks:
            block_offsets.append(current_offset)
            current_offset += len(b)
        block_offsets.append(current_offset)
        summaries = Summaries.from_summaries(
            [summarize(b, docs, energy) for b in blocks]
        )
        posting_lists[component] = (docs_in_block_order, block_offsets, summaries)
    return posting_lists


def posting_lists_to_obj(posting_lists: PostingLists) -> dict[str, Any]:
    return {k: (v[0], v[1], v[2].to_dict()) for k, v in posting_lists.items()}


def posting_lists_from_obj(obj: dict[str, Any]) -> PostingLists:
    return {k: (v[0], v[1], Summaries.from_dict(v[2])) for k, v in obj.items()}


def search(
    docs: dict[int, Entry],
    posting_lists: PostingLists,
    query: Vector,
    query_cut: int,
    heap_factor: float,
    top_k: int,
) -> list[SearchResult]:
    heap: list[tuple[float, int]] = []
    seen: set[int] = set()

    components = [
        k[0] for k in sorted(query.items(), key=lambda x: x[1], reverse=True)
    ][:query_cut]
    for c in components:
        if c not in posting_lists:
            continue
        all_blocks, block_offsets, summaries = posting_lists[c]
        dots = summaries.distances(query)
        for i in range(len(block_offsets) - 1):
            block = all_blocks[block_offsets[i] : block_offsets[i + 1]]
            if len(heap) == top_k and dots[i] < heap[0][0] * heap_factor:
                # score of the summary
                # for this block is lower
                # than the lowest score in the current top-k,
                # skip
                continue
            for doc in block:
                if doc in seen:
                    continue
                seen.add(doc)
                score = dot(docs[doc]["vector"], query)
                if len(heap) < top_k:
                    heapq.heappush(heap, (score, doc))
                elif len(heap) == top_k and score > heap[0][0]:
                    heapq.heapreplace(heap, (score, doc))
                else:
                    continue
    top_k_sorted = [heapq.heappop(heap) for el in range(len(heap))]
    top_k_sorted.reverse()

    return [{"doc_id": x[1], "score": x[0]} for x in top_k_sorted]
