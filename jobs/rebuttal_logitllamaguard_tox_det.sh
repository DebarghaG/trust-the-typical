#!/bin/bash
#SBATCH --job-name=large_scale_id_tox
#SBATCH --account=PAS2271
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH --output=logs/large_scale_id_%j.out
#SBATCH --error=logs/large_scale_id_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs


# Activate environment (adjust path as needed)
source /users/PAS2271/debargha/distribution-guardrails/env/bin/activate

# Set HF_TOKEN in your shell before running, e.g. export HF_TOKEN=hf_...

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Large-scale experiment with comprehensive in-distribution datasets and ALL baselines
echo "Starting large-scale in-distribution toxicity detection experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python experiments/unified_baselines_toxicity_detection.py \
    --methods "llamaguard_logit_vllm" \
    --datasets "realtoxicity,civil_comments,hateval,davidson,hasoc,offenseval,xsafety,polyguard_social_media,polyguard_education,polyguard_hr,polyguard_code,polyguard_cyber,rtp_alx_en" \
    --safe_dataset "id_mix" \
    --max_safe_samples 40000 \
    --max_toxic_samples 10000 \
    --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
    --batch_size 16 \
    --seed 42

echo "Large-scale ID experiment completed"
