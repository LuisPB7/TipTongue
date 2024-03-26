python3 train.py --domains all --backbone OpenMatch/co-condenser-large-msmarco
python3 inference.py --domain movie --backbone OpenMatch/co-condenser-large-msmarco
python3 index_faiss.py --domain movie
python3 maxp.py --domain movie
python3 convert_dict_to_run.py --domain movie
python3 -m pyserini.eval.trec_eval -q -m all_trec -M 1000 -m recip_rank ../DATA/test_movie_qrels.tsv run.tsv > output_results_movie.out
