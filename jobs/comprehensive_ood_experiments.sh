#!/bin/bash
#SBATCH --job-name=comprehensive_ood_tox
#SBATCH --account=PAS2271
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH --output=logs/comprehensive_ood_%j.out
#SBATCH --error=logs/comprehensive_ood_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment (adjust path as needed)
source /users/PAS2271/debargha/distribution-guardrails/env/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Comprehensive OOD experiment with ALL available toxic datasets and ALL baselines
echo "Starting comprehensive out-of-distribution toxicity detection experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python experiments/unified_baselines_toxicity_detection.py \
    --methods "MDJudge_VLLM,PolyGuard_VLLM,RMD,VIM,CIDER,fDBD,NNGuide,ReAct,GMM,AdaScale,OpenMax,Forte_GMM,Forte_OCSVM,LlamaGuard,DuoGuard" \
    --datasets "realtoxicity,civil_comments,hateval,davidson,hasoc,offenseval,xsafety,xsafety_zh,xsafety_ar,xsafety_hi,xsafety_ru,polyguard_social_media,polyguard_education,polyguard_hr,polyguard_finance_input,polyguard_finance_output,polyguard_law_input,polyguard_law_output,polyguard_regulation_input,polyguard_regulation_output,polyguard_code,polyguard_cyber,rtp_alx_en,rtp_alx_es,rtp_alx_fr,rtp_alx_de,rtp_alx_zh,rtp_alx_ja" \
    --safe_dataset "alpaca" \
    --max_safe_samples 20000 \
    --max_toxic_samples 1500 \
    --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
    --batch_size 32 \
    --seed 42

echo "Comprehensive OOD experiment completed"
