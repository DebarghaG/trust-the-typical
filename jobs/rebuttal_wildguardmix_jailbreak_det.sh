#!/bin/bash
#SBATCH --job-name=wildguardmix_jailbreak
#SBATCH --account=PAS2271
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH --output=logs/wildguardmix_jailbreak_%j.out
#SBATCH --error=logs/wildguardmix_jailbreak_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Set HF_TOKEN in your shell before running, e.g. export HF_TOKEN=hf_...

# Activate environment (adjust path as needed)
source /users/PAS2271/debargha/distribution-guardrails/env/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# WildGuardMix jailbreaking detection experiment
echo "Starting WildGuardMix jailbreaking detection experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python experiments/unified_baselines_jailbreaking_detection.py \
    --methods "RMD,VIM,CIDER,fDBD,NNGuide,ReAct,GMM,AdaScale,OpenMax,Forte_GMM,Forte_OCSVM,LlamaGuard,DuoGuard,MDJudge_VLLM,PolyGuard_VLLM" \
    --datasets "wildguardmix_train,wildguardmix_test" \
    --safe_dataset "id_mix" \
    --max_safe_samples 40000 \
    --max_adversarial_samples 10000 \
    --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
    --batch_size 16 \
    --seed 42

echo "WildGuardMix jailbreaking experiment completed"
