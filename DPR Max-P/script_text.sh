#!/bin/bash
#SBATCH --job-name=int
#SBATCH --output=output_text.out
#SBATCH -N 1 # Same machine
#SBATCH -n 12
#SBATCH --mem 64GB
#SBATCH -t 0 # unlimited time for executing


python3 convert_to_text.py
sbatch script_index.sh
