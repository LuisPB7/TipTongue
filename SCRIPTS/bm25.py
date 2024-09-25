import pickle
import pyterrier as pt
from pyterrier.measures import AP, nDCG, R, RR
from typing import List, Dict, Optional, Tuple, Union
import argparse

pt.init()


def corpus_iter(corpus_path: str):
    with open(corpus_path, "rb") as f:
        corpus = pickle.load(f)
        for docid, text in corpus.items():
            yield {"docno": docid, "text": text}


def create_index(corpus_path: str, index_path: str):
    iter_indexer = pt.IterDictIndexer(
        index_path,
        blocks=True,
        verbose=True,
        overwrite=True,
        threads=4,
    )
    index_ref = iter_indexer.index(corpus_iter(corpus_path))
    return index_ref


def title_text(title: str, text: str) -> str:
    return "".join([c if c.isalnum() else " " for c in (title + " " + text)])


def load_queries(queries_path: str, titles_path: str):
    with open(queries_path, "rb") as f:
        queries = pickle.load(f)

    with open(titles_path, "rb") as f:
        titles = pickle.load(f)

    queries_title_text = {qid: title_text(titles[qid], queries[qid]) for qid in queries}
    return queries_title_text


def load_qrels(qrels_path: str):
    with open(qrels_path, "rb") as f:
        qrels = pickle.load(f)
    return qrels


def run_bm25(
    corpus_path: str,
    index_path: str,
    queries_path: str,
    titles_path: str,
    qrels_path: str,
    domain: str = "",
    results_dir: Optional[str] = None,
):
    index_ref = create_index(corpus_path, index_path)
    queries = load_queries(queries_path, titles_path)
    qrels = load_qrels(qrels_path)

    BM25 = pt.BatchRetrieve(
        index_ref,
        wmodel="BM25",
        verbose=True,
        controls={"bm25.b": 1, "bm25.k_1": 0.8},
    )

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
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus")
    parser.add_argument("--index_path", type=str, required=True, help="Path to save the index")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries, should be a pickled dictionary of qid -> query")
    parser.add_argument("--titles_path", type=str, required=True, help="Path to the titles, should be a pickled dictionary of qid -> title")
    parser.add_argument("--qrels_path", type=str, required=True, help="Path to the qrels, should be a pickled dictionary of qid -> docid")
    parser.add_argument("--domain", type=str, required=True, help="Domain of the queries", choices=["movie", "book", "music", "game"])
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save the results")

    args = parser.parse_args()

    results = run_bm25(
        corpus_path=args.corpus_path,
        index_path=args.index_path,
        queries_path=args.queries_path,
        titles_path=args.titles_path,
        qrels_path=args.qrels_path,
        domain=args.domain,
        results_dir=args.results_dir,
    )
    results.style.set_precision(3)
    print(results)

    results.to_csv(f"{args.results_dir}/{args.domain}_results.tsv", sep="\t")
