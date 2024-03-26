python3 chatgpt.py --domain movie
python3 search_again.py --domain movie
python3 convert_dict_to_run.py --domain movie
python3 -m pyserini.eval.trec_eval -q -m all_trec -M 1000 -m recip_rank ../../DATA/test_movie_qrels.tsv run.tsv > output_results.out
