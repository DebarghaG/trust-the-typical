#!/bin/bash
#SBATCH --job-name=lightweight_baselines_tox
#SBATCH --account=PAS2271
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#SBATCH --output=logs/lightweight_baselines_%j.out
#SBATCH --error=logs/lightweight_baselines_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs


# Activate environment (adjust path as needed)
source /users/PAS2271/debargha/distribution-guardrails/env/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Lightweight baselines comparison (non-VLLM methods only for faster iteration)
echo "Starting lightweight baselines comparison toxicity detection experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python experiments/unified_baselines_toxicity_detection.py \
    --methods "RMD,VIM,CIDER,fDBD,NNGuide,ReAct,GMM,AdaScale,OpenMax,Forte_GMM,Forte_OCSVM" \
    --datasets "realtoxicity,civil_comments,hateval,davidson,hasoc,offenseval,xsafety,polyguard_social_media,polyguard_education,polyguard_code,rtp_alx_en,rtp_alx_zh,rtp_alx_es" \
    --safe_dataset "id_mix" \
    --max_safe_samples 35000 \
    --max_toxic_samples 1500 \
    --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
    --batch_size 32 \
    --seed 42

echo "Lightweight baselines comparison experiment completed"
