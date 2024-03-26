#!/bin/bash
#SBATCH --job-name=dpr
#SBATCH --output=output.out
#SBATCH --nodes 1 # Same machine
#SBATCH --ntasks-per-node 4
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --mem 64GB
#SBATCH -t 0 # unlimited time for executing

module load anaconda3
module load cuda-10.0
module load cudnn-10.0-7.3

source activate myenv


srun python3 train.py
python3 split_documents.py
python3 inference.py
python3 index_faiss.py
python3 convert_dict_to_run.py
python3 -m pyserini.eval.trec_eval -q -m all_trec -M 10 -m recip_rank ./data/dev_qrel.txt run.tsv
