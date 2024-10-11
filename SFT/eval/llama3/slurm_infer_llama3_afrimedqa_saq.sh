#!/bin/bash

#SBATCH --job-name="llama3_evals"

#SBATCH --output=/srv/essa-lab/flash3/cnimo3/work/reproduction/llama3_evals_logs/sample-%j.out

#SBATCH --error=/srv/essa-lab/flash3/cnimo3/work/reproduction/llama3_evals_logs/sample-%j.err

## number of nodes
#SBATCH --nodes=1

## number of tasks per node

#SBATCH --ntasks-per-node=1

#SBATCH --gpus=a40:2

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

chmod +x "/srv/essa-lab/flash3/cnimo3/work/reproduction/PMC-LLaMA/SFT/eval/saq_inference_pipeline.sh"

srun /srv/essa-lab/flash3/cnimo3/work/reproduction/PMC-LLaMA/SFT/eval/saq_inference_pipeline.sh -m /srv/essa-lab/flash3/cnimo3/work/reproduction/Meta-Llama-3-8B-Instruct  -d /srv/essa-lab/flash3/cnimo3/work/reproduction/AfriMedQA/AfriMed-QA/data/afri_med_qa_saq.jsonl -i /srv/essa-lab/flash3/cnimo3/work/reproduction/AfriMedQA/AfriMed-QA/my_results/saq/llama3_7b_result_dir_2 -p saq -n 0