#!/bin/bash
#SBATCH --job-name=domain_specific_tox
#SBATCH --account=PAS2271
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00

#SBATCH --output=logs/domain_specific_%j.out
#SBATCH --error=logs/domain_specific_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs



# Activate environment (adjust path as needed)
source /users/PAS2271/debargha/distribution-guardrails/env/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Domain-specific experiment focusing on PolyGuard domains
echo "Starting domain-specific toxicity detection experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python experiments/unified_baselines_toxicity_detection.py \
    --methods "MDJudge_VLLM,PolyGuard_VLLM,RMD,VIM,CIDER,fDBD,NNGuide,ReAct,GMM,AdaScale,OpenMax,Forte_GMM,Forte_OCSVM,LlamaGuard,DuoGuard" \
    --datasets "polyguard_social_media,polyguard_education,polyguard_hr,polyguard_finance_input,polyguard_finance_output,polyguard_law_input,polyguard_law_output,polyguard_regulation_input,polyguard_regulation_output,polyguard_code,polyguard_cyber" \
    --safe_dataset "dolly" \
    --max_safe_samples 35000 \
    --max_toxic_samples 1200 \
    --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
    --batch_size 32 \
    --seed 42

echo "Domain-specific experiment completed"
