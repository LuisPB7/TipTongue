import pickle
import pyterrier as pt
from pyterrier.measures import AP, nDCG, R, RR
from typing import List, Dict, Optional, Tuple, Union
import argparse
import os
from tqdm import tqdm
import ujson as json
import pandas as pd

pt.init()


def corpus_iter(corpus_dir: str):
    for shard in tqdm(os.listdir(corpus_dir)):
        with open(os.path.join(corpus_dir, shard), "r") as f:
            for line in f:  # jsonl
                doc = json.loads(line)
                for docid, text in doc.items():
                    yield {"docno": docid, "text": text}


def create_index(corpus_dir: str, index_dir: str):
    print("Creating index")
    iter_indexer = pt.IterDictIndexer(
        index_dir,
        blocks=True,
        verbose=True,
        overwrite=True,
        threads=24,
    )
    index_ref = iter_indexer.index(corpus_iter(corpus_dir))
    print("Index created")
    return index_ref


def title_text(title: str, text: str) -> str:
    return "".join([c if c.isalnum() else " " for c in (title + " " + text)])


def load_queries(queries_path: str, titles_path: str) -> pd.DataFrame:
    with open(queries_path, "rb") as f:
        queries = pickle.load(f)

    with open(titles_path, "rb") as f:
        titles = pickle.load(f)

    queries_title_text = {qid: title_text(titles[qid], queries[qid]) for qid in queries}
    return pd.DataFrame(
        [{"qid": qid, "query": query} for qid, query in queries_title_text.items()],
        columns=["qid", "query"],
    )


def load_qrels(qrels_path: str) -> pd.DataFrame:
    with open(qrels_path, "rb") as f:
        qrels = pickle.load(f)
    return pd.DataFrame(
        [{"qid": qid, "docno": docno, "label": 1} for qid, docno in qrels.items()],
        columns=["qid", "docno", "label"],
    )


def run_bm25(
    corpus_dir: str,
    index_dir: str,
    queries_path: str,
    titles_path: str,
    qrels_path: str,
    domain: str = "",
    results_dir: Optional[str] = None,
):
    if os.path.exists(os.path.join(index_dir)):
        index_ref = pt.IndexFactory.of(index_dir, memory=True)
    else:
        index_ref = create_index(corpus_dir, index_dir)

    queries = load_queries(queries_path, titles_path)
    qrels = load_qrels(qrels_path)

    # filter queries to only those in qrels
    qids = set(qrels["qid"])
    queries = queries[queries["qid"].isin(qids)]


    BM25 = pt.terrier.Retriever(
        index_ref,
        wmodel="BM25",
        verbose=True,
        controls={"bm25.b": 1, "bm25.k_1": 0.8},
    )

    print("Running BM25")

    results = pt.Experiment(
        [BM25],
        queries,
        qrels,
        eval_metrics=[
            "map",
            "recip_rank",
            R @ 1,
            R @ 10,
            R @ 100,
            R @ 1000,
            nDCG @ 10,
            nDCG @ 100,
            nDCG @ 1000,
        ],
        names=[f"BM25_{domain}"],
        save_dir=results_dir,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_dir",
        type=str,
        required=True,
        help="Path to the corpus, jsonl shards of {docid: text}",
    )
    parser.add_argument(
        "--index_dir", type=str, required=True, help="Path to save the index"
    )
    parser.add_argument(
        "--queries_path",
        type=str,
        required=True,
        help="Path to the queries, should be a pickled dictionary of qid -> query",
    )
    parser.add_argument(
        "--titles_path",
        type=str,
        required=True,
        help="Path to the titles, should be a pickled dictionary of qid -> title",
    )
    parser.add_argument(
        "--qrels_path",
        type=str,
        required=True,
        help="Path to the qrels, should be a pickled dictionary of qid -> docid",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain of the queries",
        choices=["movie", "book", "music", "game"],
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Directory to save the results"
    )

    args = parser.parse_args()

    results = run_bm25(
        corpus_dir=args.corpus_dir,
        index_dir=args.index_dir,
        queries_path=args.queries_path,
        titles_path=args.titles_path,
        qrels_path=args.qrels_path,
        domain=args.domain,
        results_dir=args.results_dir,
    )
    print(results)

    results.to_csv(f"{args.results_dir}/{args.domain}_results.tsv", sep="\t")
