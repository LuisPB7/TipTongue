#!/bin/bash
#SBATCH --job-name=ae_metrics
#SBATCH --output=output_results.out
#SBATCH -N 1 # Same machine
#SBATCH -n 1
#SBATCH --mem 8GB
#SBATCH -t 0 # unlimited time for executing

TREC=19

python3 -m pyserini.eval.trec_eval -m all_trec -M 10 -q -m recip_rank ../../MSMARCO-PASSAGE/data/qrels.dev.small.tsv run.tsv
