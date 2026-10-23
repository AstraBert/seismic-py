# seismic-py

A small Python re-implementation of [Seismic](https://github.com/TusKANNy/seismic), an approximate search method for learned sparse vectors (for example SPLADE output).

This project is for learning. It is written in plain Python with no numerical libraries. It is easy to read and slow to run. Do not use it for production search.

## What it does

A sparse vector has many dimensions, and almost all of them are zero. A document is stored as a list of (token, weight) pairs. Given a query, the goal is to find the documents with the highest dot product with the query.

Exact search scores every document. Seismic scores far fewer of them. It does this in two phases.

At build time, for each token:

1. **Prune.** Keep only the strongest (token, document) pairs. These form the posting list of the token.
2. **Block.** Group the documents of each posting list into blocks of similar documents, using random k-means.
3. **Summarize.** Give each block a short summary vector. A summary holds the largest weight of each token in the block, cut down to the strongest entries. Summaries are stored with 8-bit values.

At search time:

1. Take the `query_cut` query tokens with the largest weights, and visit their posting lists.
2. Score the summary of each block against the query. This is cheap.
3. Skip the block if its summary score is below `heap_factor` times the lowest score in the current top-k.
4. Otherwise, score the documents of the block exactly, and keep the best `k`.

The result is approximate. Pruning and block skipping can drop a good document, and the settings below control this trade-off.

## Requirements

- Python 3.14 or later
- [uv](https://docs.astral.sh/uv/)

## Quick start

The commands below build a fake collection, index it, and search it. They need no outside data.

```bash
uv sync

# 1. generate 3,000 documents and 100 queries (written to ./data)
uv run seismic-py synth --output data

# 2. build the index
uv run seismic-py index data/documents.jsonl --output index.bin

# 3. search with every query
uv run seismic-py search index.bin data/queries.jsonl --output results.json
```

Run `uv run seismic-py --help`, or add `--help` to any command, for the full list of options.

## Input format

One JSON object per line (JSONL), the same format that Seismic uses:

```json
{"id": 0, "content": "optional text", "vector": {"token_a": 0.91, "token_b": 0.17}}
```

- `id`: an integer, unique in the file.
- `content`: optional text. The index does not use it.
- `vector`: a dictionary from token (string) to weight (number). Use positive weights. An entry with a weight of 0 or less never gets a posting, so the search cannot find a document through it.

Queries use the same format.

## Output format

`search` writes one JSON object. Each key is a query id. Each value is the list of results, best first:

```json
{
  "0": [
    {"doc_id": 1731, "score": 5.94383154},
    {"doc_id": 1385, "score": 5.2502767}
  ]
}
```

## Commands and options

### `synth`

Makes fake data with a Zipf-like token distribution and a topic structure, so that blocks have something to find.

| Option | Default | Meaning |
|---|---|---|
| `--n-documents` | 3000 | Number of documents |
| `--n-queries` | 100 | Number of queries |
| `--dim` | 300 | Number of distinct tokens |
| `--output` | `output` | Directory for `documents.jsonl` and `queries.jsonl` |

### `index`

Usage: `seismic-py index DOCUMENTS_PATH [OPTIONS]`

| Option | Default | Meaning |
|---|---|---|
| `--output` | `index.bin` | Index file to write |
| `--n-postings` | 3500 | Average number of documents kept per token. About `n-postings` times the number of tokens postings are kept in total. Higher: better recall, bigger index, slower build and search. |
| `--max-fraction` | 1.5 | A single posting list can hold at most `n-postings` times this many documents. It stops very common tokens from taking too many postings. |
| `--centroid-fraction` | 0.3 | Share of a posting list's documents that become k-means centroids. It sets the number of blocks. |
| `--min-cluster-size` | 2 | Clusters with this many documents or fewer are dissolved, and their documents join other clusters. |
| `--summary-energy` | 0.4 | Fraction (0 to 1) of a summary's total weight to keep. Lower: smaller summaries, faster block scoring, but a summary can underestimate its block. `1.0` keeps every entry. |
| `--seed` | 42 | Seed for the random choice of centroids |

### `search`

Usage: `seismic-py search INDEX_FILE QUERIES_FILE [OPTIONS]`

| Option | Default | Meaning |
|---|---|---|
| `--output` | `output.json` | File for the results |
| `--n-queries` | all | Search a random sample of this many queries |
| `--top-k` | 10 | Number of results per query |
| `--query-cut` | 10 | Visit only the posting lists of the N query tokens with the largest weights. Higher: better recall, slower. |
| `--heap-factor` | 0.7 | Skip a block when its summary score is below this factor times the lowest score in the current top-k. Higher: more blocks skipped, faster, lower recall. `0` never skips. |

## Measuring recall

There is no command for this. Compare the results with an exact search, using the helpers in `seismic_py.data`:

```python
import json
from seismic_py.data import brute_force_on_ids, load_data, recall

docs, token_map = load_data("data/documents.jsonl")
queries, _ = load_data("data/queries.jsonl", token_map)
results = json.load(open("results.json"))

scores = []
for qid, q in queries.items():
    exact = brute_force_on_ids(docs, q["vector"], 10)
    r = recall(exact, [hit["doc_id"] for hit in results[str(qid)]])
    if r is not None:  # None means no document shares a token with the query
        scores.append(r)
print(f"recall@10 = {sum(scores) / len(scores):.3f} over {len(scores)} queries")
```

Recall at 10 is the share of the exact top 10 that the approximate search also returns.

## Example numbers

These come from one run on the fake data of `synth` with its defaults (3,000 documents, 100 queries, 300 tokens). Real data behaves differently, so use them only as a rough guide.

| Index | `--query-cut` | `--heap-factor` | recall@10 |
|---|---|---|---|
| defaults | 10 | 0.0 | 1.000 |
| defaults | 10 | 0.7 | 0.998 |
| defaults | 5 | 0.9 | 0.993 |
| `--n-postings 60 --max-fraction 2 --centroid-fraction 0.1` | 10 | 0.7 | 0.997 |
| the same pruned index | 2 | 0.9 | 0.928 |

Building took about 7 seconds with the defaults, and about 0.2 seconds with the pruned settings.

With the default `--n-postings 3500` and a small collection, pruning does nothing. The number of postings to keep is more than the number that exist. To see pruning, lower `--n-postings`.

## How the index file is stored

The index is one [MessagePack](https://msgpack.org) file with three parts:

- `docs`: all documents (the "forward index" used for exact scoring),
- `token_map`: token to integer id,
- `posting_lists`: for each token, the document ids in block order, the block offsets, and the block summaries.

The file contains the full documents, so it is larger than the posting lists alone.

## Code layout

```text
src/seismic_py/
├── __init__.py       command-line interface (typer): synth, index, search
├── data.py           JSONL loading, dot product, brute force, recall, fake data
└── mini_seismic.py   pruning, blocking, summaries, index build, search
```

Main functions in `mini_seismic.py`:

| Function | Job |
|---|---|
| `global_threshold_pruning` | Choose which (token, document) pairs get a posting |
| `kmeans_blocks` | Group the documents of one posting list into blocks |
| `summarize` | Build the summary vector of one block |
| `Summaries` | Store the summaries of one posting list (8-bit values), and score them against a query |
| `build_posting_lists` | Run the build for every token |
| `search` | Visit posting lists, skip blocks, score documents, keep the top k |

## Differences from the real Seismic

- Vectors are Python dictionaries. Seismic stores them in compact arrays.
- k-means is exact: each document is compared with every centroid. Seismic uses an approximate version by default.
- Each centroid always forms the start of its own cluster. Seismic assigns every document, centroids included, to the closest centroid.
- Pruning does not trim documents that tie at the threshold.
- There is no kNN graph and no `first_sorted` option.
- Posting lists hold document ids. Seismic stores packed positions in its forward index.
- No compression of posting lists, no f16 or fixed-point values for documents, and no parallelism.
- Default `--centroid-fraction` is 0.3 here. The Seismic default is 0.1.

## Development

```bash
uv run ruff format
uv run ruff check --fix
uv run ty check src/
```

The same tasks are in `jakefile.toml`.

## References

Seismic: Bruch, Nardini, Rulli, Venturini. *Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations.* ACM SIGIR 2024. The original code and further papers are at <https://github.com/TusKANNy/seismic>.
