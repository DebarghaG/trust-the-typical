#!/bin/bash
#SBATCH --job-name=embedding_ablation_tox
#SBATCH --account=PAS2271
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00

#SBATCH --output=logs/embedding_ablation_%j.out
#SBATCH --error=logs/embedding_ablation_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs


# Activate environment (adjust path as needed)
source /users/PAS2271/debargha/distribution-guardrails/env/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Embedding model ablation study with E5 model
echo "Starting embedding model ablation toxicity detection experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python experiments/unified_baselines_toxicity_detection.py \
    --methods "MDJudge_VLLM,PolyGuard_VLLM,RMD,VIM,CIDER,fDBD,NNGuide,ReAct,GMM,AdaScale,OpenMax,Forte_GMM,Forte_OCSVM,LlamaGuard,DuoGuard" \
    --datasets "realtoxicity,civil_comments,hateval,davidson,xsafety,polyguard_social_media,rtp_alx_en" \
    --safe_dataset "id_mix" \
    --max_safe_samples 25000 \
    --max_toxic_samples 1500 \
    --embedding_model "intfloat/e5-large-v2" \
    --batch_size 24 \
    --seed 42

echo "Embedding model ablation experiment completed"
