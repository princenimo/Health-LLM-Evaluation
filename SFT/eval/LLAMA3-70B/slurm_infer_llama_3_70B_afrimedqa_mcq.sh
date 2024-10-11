#!/bin/bash

#SBATCH --job-name="ML70TAM"

#SBATCH --output=/srv/essa-lab/flash3/cnimo3/work/reproduction/ML70TAM_logs/sample-%j.out

#SBATCH --error=/srv/essa-lab/flash3/cnimo3/work/reproduction/ML70TAM_logs/sample-%j.err

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

chmod +x "/srv/essa-lab/flash3/cnimo3/work/reproduction/PMC-LLaMA/SFT/eval/mcq_inference_pipeline.sh"

srun /srv/essa-lab/flash3/cnimo3/work/reproduction/PMC-LLaMA/SFT/eval/mcq_inference_pipeline.sh -m  /srv/essa-lab/flash3/cnimo3/work/reproduction/Meta-Llama-3-70B-Instruct  -d /srv/essa-lab/flash3/cnimo3/work/reproduction/PMC-LLaMA/SFT/eval/Traditional_African_Medicine_MCQA.jsonl -i /srv/essa-lab/flash3/cnimo3/work/reproduction/TAM_LLAMA3_results/Meta-Llama-3-70B -p mcqinstruct -n 0