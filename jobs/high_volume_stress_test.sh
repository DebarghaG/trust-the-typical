#!/bin/bash
#SBATCH --job-name=high_volume_stress_tox
#SBATCH --account=PAS2271
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00

#SBATCH --output=logs/high_volume_stress_%j.out
#SBATCH --error=logs/high_volume_stress_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment (adjust path as needed)
source /users/PAS2271/debargha/distribution-guardrails/env/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# High-volume stress test with maximum samples for robustness evaluation
echo "Starting high-volume stress test toxicity detection experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python experiments/unified_baselines_toxicity_detection.py \
    --methods "MDJudge_VLLM,PolyGuard_VLLM,RMD,VIM,CIDER,fDBD,NNGuide,ReAct,GMM,AdaScale,OpenMax,Forte_GMM,Forte_OCSVM,LlamaGuard,DuoGuard" \
    --datasets "realtoxicity,civil_comments,hateval,davidson,hasoc,offenseval,xsafety,polyguard_social_media,rtp_alx_en" \
    --safe_dataset "id_mix" \
    --max_safe_samples 50000 \
    --max_toxic_samples 3000 \
    --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
    --batch_size 32 \
    --seed 42

echo "High-volume stress test experiment completed"
