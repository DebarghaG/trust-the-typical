#!/bin/bash
#SBATCH --job-name=multilingual_ablation_tox
#SBATCH --account=PAS2271
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00

#SBATCH --output=logs/multilingual_ablation_%j.out
#SBATCH --error=logs/multilingual_ablation_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs



# Activate environment (adjust path as needed)
source /users/PAS2271/debargha/distribution-guardrails/env/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Multilingual ablation study focusing on language-specific performance
echo "Starting multilingual ablation toxicity detection experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python experiments/unified_baselines_toxicity_detection.py \
    --methods "MDJudge_VLLM,PolyGuard_VLLM,RMD,VIM,CIDER,fDBD,NNGuide,ReAct,GMM,AdaScale,OpenMax,Forte_GMM,Forte_OCSVM,LlamaGuard,DuoGuard" \
    --datasets "xsafety_en,xsafety_zh,xsafety_ar,xsafety_sp,xsafety_fr,xsafety_de,xsafety_ja,xsafety_hi,xsafety_ru,rtp_alx_en,rtp_alx_zh,rtp_alx_es,rtp_alx_fr,rtp_alx_de,rtp_alx_ja,rtp_alx_ru,rtp_alx_hi" \
    --safe_dataset "openassistant" \
    --max_safe_samples 30000 \
    --max_toxic_samples 1000 \
    --embedding_model "BAAI/bge-m3" \
    --batch_size 32 \
    --seed 42

echo "Multilingual ablation experiment completed"
