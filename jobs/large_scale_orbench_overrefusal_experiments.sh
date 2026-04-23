#!/bin/bash
#SBATCH --job-name=large_scale_orbench
#SBATCH --account=PAS2271
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH --output=logs/large_scale_orbench_%j.out
#SBATCH --error=logs/large_scale_orbench_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs


# Activate environment (adjust path as needed)
source /users/PAS2271/debargha/distribution-guardrails/env/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Large-scale OR-Bench overrefusal detection experiment with ALL baselines
echo "Starting large-scale OR-Bench overrefusal detection experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python experiments/unified_baselines_orbench_overrefusal_detection.py \
    --methods "MDJudge_VLLM,RMD,VIM,CIDER,fDBD,NNGuide,ReAct,GMM,AdaScale,OpenMax,Forte_GMM,Forte_OCSVM,LlamaGuard,DuoGuard,PolyGuard_VLLM" \
    --max_safe_samples 40000 \
    --max_toxic_samples 10000 \
    --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
    --batch_size 32 \
    --seed 42

echo "Large-scale OR-Bench overrefusal experiment completed"
