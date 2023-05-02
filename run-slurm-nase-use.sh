#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH -G2
#SBATCH --mem-per-gpu=28G
#SBATCH --time=01:30:00
#SBATCH --output=logs-use/nlp-use-%J.out
#SBATCH --error=logs-use/nlp-use-%J.err
#SBATCH --job-name="NLP paraphrase training"

# train the model
# srun singularity exec --nv ./containers/nlp_pytorch.sif python "./paraphrase_projekt.py"

# just use the model 
srun singularity exec --nv ./containers/nlp_pytorch.sif python "./use_model.py"