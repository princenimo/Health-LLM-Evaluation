#!/bin/bash

#SBATCH --job-name="mixtMed"

#SBATCH --output=/srv/essa-lab/flash3/cnimo3/work/reproduction/mixtMed_medqa_logs/sample-%j.out

#SBATCH --error=/srv/essa-lab/flash3/cnimo3/work/reproduction/mixtMed_medqa_logs/sample-%j.err

## number of nodes
#SBATCH --nodes=1

## number of tasks per node

#SBATCH --ntasks-per-node=1

#SBATCH --gpus=a40:4

#SBATCH --cpus-per-task=32

#SBATCH -p essa-lab

#SBATCH --qos=short
source /nethome/cnimo3/.bashrc
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "/nethome/cnimo3/" | tr '\n' ':')
echo "--------------------"
echo $PATH
echo "--------------------"
WANDB_API_KEY="a2f43cdc0ae21e0282a5872653cb2beb89b69301"
WANDB_NAME="Meditron MedMCQA 7b"
source /srv/essa-lab/flash3/cnimo3/miniconda3/bin/activate meditron
echo "--------------------"
echo "env below"
printenv
echo "--------------------"
echo "Print Path again..just checking :)"
echo $PATH
echo "--------------------"

cd /srv/essa-lab/flash3/cnimo3/work/reproduction/meditron

srun python /srv/essa-lab/flash3/cnimo3/work/reproduction/PMC-LLaMA/SFT/eval/qa_inference.py  --model-name-or-path /srv/essa-lab/flash3/cnimo3/work/reproduction/Mixtral-8x7B-Instruct-v0.1 --data-path /srv/essa-lab/flash3/cnimo3/work/reproduction/MedQA-USMLE-4-options/phrases_no_exclude_test.jsonl --write-dir /srv/essa-lab/flash3/cnimo3/work/reproduction/medqa_test/Mixtral_7b_result_dir