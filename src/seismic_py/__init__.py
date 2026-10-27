import json
import random
from pathlib import Path
from typing import Annotated

import msgpack
import typer

from .data import Entry, load_data, make_fake_data
from .mini_grip import DEFAULT_D, DEFAULT_L, GripIndex, build_index
from .mini_grip import grip_search as ggrip_search
from .mini_seismic import (
    PostingLists,
    build_posting_lists,
    posting_lists_from_obj,
    posting_lists_to_obj,
)
from .mini_seismic import (
    search as seismic_search,
)

app = typer.Typer()


@app.command(name="index", help="Create a SEISMIC-like index for your documents")
def index(
    documents_path: Annotated[
        str,
        typer.Argument(
            help='Path to the JSONL file containing documents in the format of {"id": int, "content": string, "vector": {"<token>": float}}'
        ),
    ],
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help="Output file where to store the index.",
        ),
    ] = "index.bin",
    n_postings: Annotated[
        int,
        typer.Option(
            "--n-postings",
            help="Average number of documents kept per component (posting list). "
            "Only the strongest entries survive: about N x (number of components) "
            "postings are kept in total, across the whole collection. "
            "Higher: better recall, bigger index, slower build and search. ",
        ),
    ] = 3500,
    max_fraction: Annotated[
        float,
        typer.Option(
            "--max-fraction",
            help="Upper limit for the length of a single posting list, as a multiple "
            "of --n-postings (limit = n_postings x max_fraction). "
            "Stops very common components from taking too many postings. ",
        ),
    ] = 1.5,
    centroid_fraction: Annotated[
        float,
        typer.Option(
            "--centroid-fraction",
            help="Fraction of documents in a posting list that should be considered as centroids for clustering and summarization",
        ),
    ] = 0.3,
    min_cluster_size: Annotated[
        int,
        typer.Option(
            "--min-cluster-size",
            help="Minimum number of documents in a cluster for it not to be merged with larger ones",
        ),
    ] = 2,
    energy: Annotated[
        float,
        typer.Option(
            "--summary-energy",
            help="How much of each block summary to keep, as a fraction (0 to 1) of "
            "the summary's total weight. The summary keeps its largest entries "
            "until their values add up to E x total, and drops the rest. "
            "Lower: smaller index, faster block scoring, but summaries can "
            "underestimate a block, so good blocks may be skipped. ",
        ),
    ] = 0.4,
    seed: Annotated[
        int, typer.Option("--seed", help="Seed for random centroid sampling")
    ] = 42,
) -> None:
    docs, token_to_id_map = load_data(documents_path)
    rng = random.Random(seed)
    posting_lists = build_posting_lists(
        docs,
        dim=len(token_to_id_map),
        n_postings=n_postings,
        max_fraction=max_fraction,
        centroid_fraction=centroid_fraction,
        min_cluster_size=min_cluster_size,
        rng=rng,
        energy=energy,
    )
    to_pack = {
        "docs": docs,
        "token_map": token_to_id_map,
        "posting_lists": posting_lists_to_obj(posting_lists),
    }
    packed = msgpack.packb(to_pack)
    if packed is None:
        typer.echo("Packing did not produce any bytes", color=True, err=True)
        raise typer.Exit(1)
    with open(output, "wb") as f:
        f.write(packed)


@app.command(
    name="synth",
    help="Generate synthetic documents and queries that follow a Zipfian distribution",
)
def synth(
    n_documents: Annotated[
        int, typer.Option("--n-documents", help="Number of documents to generate")
    ] = 3000,
    n_queries: Annotated[
        int, typer.Option("--n-queries", help="Number of queries to generate")
    ] = 100,
    dim: Annotated[
        int, typer.Option("--dim", help="Number of tokens in the vocabulary")
    ] = 300,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help="Directory where to store queries.jsonl and documents.jsonl",
        ),
    ] = "output",
) -> None:
    docs, queries = make_fake_data(
        n_docs=n_documents, n_queries=n_queries, dim=dim, as_raw=True
    )
    p = Path(output)
    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
    with open(p / "documents.jsonl", "w") as f:
        f.writelines([json.dumps(d) + "\n" for d in docs])
    with open(p / "queries.jsonl", "w") as f:
        f.writelines([json.dumps(d) + "\n" for d in queries])


@app.command(name="grip-build", help="Build a grip index")
def grip_build(
    documents_path: Annotated[
        str,
        typer.Argument(
            help='Path to the JSONL file containing documents in the format of {"id": int, "content": string, "vector": {"<token>": float}}'
        ),
    ],
    graph_d: Annotated[int, typer.Option("-d", "--graph-d")] = DEFAULT_D,
    seed_l: Annotated[int, typer.Option("-l", "--seed-l")] = DEFAULT_L,
    sketch_r: Annotated[int, typer.Option("-r", "--sketch-r")] = 8,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help="File where to store the grip index",
        ),
    ] = "index.grip",
) -> None:
    index = build_index(documents_path, graph_d, seed_l, sketch_r)
    cont = msgpack.packb(index)
    if cont is None:
        raise RuntimeError("Could not pack index in bytes")
    with open(output, "wb") as o:
        o.write(cont)


@app.command(name="grip-search", help="Build a grip index")
def grip_search(
    index_path: Annotated[
        str,
        typer.Argument(help="Path to the file containing the grip index"),
    ],
    queries_path: Annotated[
        str,
        typer.Option("-q", "--queries", help="Path to the file containing the queries"),
    ],
    queries: Annotated[
        list[int] | None, typer.Option("--query-index", help="Query indices to use for search")
    ] = None,
    ef: Annotated[int, typer.Option("--ef")] = 10,
    k: Annotated[int, typer.Option("--top-k")] = 10,
    use_seeds: Annotated[bool, typer.Option("--use-seeds")] = False,
    use_sketches: Annotated[bool, typer.Option("--use-sketches")] = False,
    entrypoint_t: Annotated[int, typer.Option("--top-k")] = 10,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help="File where to store the search output",
        ),
    ] = "output.json",
) -> None:
    with open(index_path, "rb") as f:
        content = f.read()
    index: GripIndex = msgpack.unpackb(content, strict_map_key=False)
    quer, _ = load_data(queries_path)
    results = {}
    queries = [k for k in quer] if queries is None else queries
    for q in queries:
        result = ggrip_search(
            index, quer[q]["vector"], ef, use_seeds, use_sketches, k, entrypoint_t
        )
        results[q] = result
    with open(output, "w") as f:
        json.dump(results, f, indent=2)


@app.command(name="search", help="Search the index with a query")
def search(
    index_file: Annotated[
        str, typer.Argument(help="File where to load the index from")
    ],
    queries_file: Annotated[
        str, typer.Argument(help="File where to load the queries from")
    ],
    output: Annotated[
        str, typer.Option("--output", help="JSON file where to store retrieval results")
    ] = "output.json",
    num_queries: Annotated[
        int | None,
        typer.Option(
            "--n-queries",
            help="Number of queries to sample. Defaults to using all queries",
        ),
    ] = None,
    top_k: Annotated[
        int, typer.Option("--top-k", help="Top-K matches to retrieve for each query")
    ] = 10,
    query_cut: Annotated[
        int,
        typer.Option(
            "--query-cut",
            help="Number of top query components to decide which posting lists to visit",
        ),
    ] = 10,
    heap_factor: Annotated[
        float,
        typer.Option(
            "--heap-factor",
            help="Skip a block of documents when its summary score is below "
            "F x (the lowest score in the current top-k). "
            "Higher: more blocks skipped, faster, lower recall. ",
        ),
    ] = 0.7,
) -> None:
    with open(index_file, "rb") as f:
        index = msgpack.unpackb(f.read(), strict_map_key=False)
    docs: dict[int, Entry] = index["docs"]
    token_map: dict[str, int] = index["token_map"]
    posting_list: PostingLists = posting_lists_from_obj(index["posting_lists"])
    queries, token_map = load_data(queries_file, token_to_id_map=token_map)
    if num_queries is not None:
        to_query = random.sample(list(queries.keys()), k=min(len(queries), num_queries))
    else:
        to_query = list(queries.keys())
    results = {}
    for q in to_query:
        res = seismic_search(
            docs,
            posting_lists=posting_list,
            query=queries[q]["vector"],
            query_cut=query_cut,
            heap_factor=heap_factor,
            top_k=top_k,
        )
        results[q] = res
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
