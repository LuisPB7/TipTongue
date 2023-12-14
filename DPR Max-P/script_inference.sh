
CUDA_VISIBLE_DEVICES=0 python3 rerank.py --batch_size 4096 --vocab_size 98304
python3 convert_dict_to_run.py
python3 -m pyserini.eval.trec_eval -m all_trec -M 10 -q -m recip_rank ../../MSMARCO-PASSAGE/data/qrels.dev.small.tsv run.tsv > output_inference.out
