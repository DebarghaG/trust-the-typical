#!/bin/bash
#SBATCH --job-name=large_scale_jailbreak
#SBATCH --account=PAS2271
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH --output=logs/large_scale_jailbreak_%j.out
#SBATCH --error=logs/large_scale_jailbreak_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Set HF_TOKEN in your shell before running, e.g. export HF_TOKEN=hf_...

# Activate environment (adjust path as needed)
source /users/PAS2271/debargha/distribution-guardrails/env/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Large-scale experiment with comprehensive adversarial datasets and ALL baselines
echo "Starting large-scale jailbreaking detection experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python experiments/unified_baselines_jailbreaking_detection.py \
    --methods "energy,knn,mahalanobis" \
    --datasets "advbench,harmbench,harmbench_standard,harmbench_contextual,harmbench_copyright,jailbreakbench,maliciousinstruct,xstest,beavertails,sorrybench" \
    --safe_dataset "id_mix" \
    --max_safe_samples 40000 \
    --max_adversarial_samples 10000 \
    --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
    --batch_size 16 \
    --seed 42

echo "Large-scale jailbreaking experiment completed"
