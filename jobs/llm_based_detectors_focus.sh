#!/bin/bash
#SBATCH --job-name=llm_detectors_tox
#SBATCH --account=PAS2271
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00

#SBATCH --output=logs/llm_detectors_%j.out
#SBATCH --error=logs/llm_detectors_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs


# Activate environment (adjust path as needed)
source /users/PAS2271/debargha/distribution-guardrails/env/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# LLM-based detectors focused experiment
echo "Starting LLM-based detectors focused toxicity detection experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python experiments/unified_baselines_toxicity_detection.py \
    --methods "MDJudge_VLLM,PolyGuard_VLLM,LlamaGuard,DuoGuard,Forte_GMM,Forte_OCSVM" \
    --datasets "realtoxicity,civil_comments,hateval,davidson,hasoc,offenseval,xsafety,xsafety_zh,xsafety_ar,xsafety_hi,polyguard_social_media,polyguard_education,polyguard_hr,polyguard_finance_input,polyguard_law_input,polyguard_code,polyguard_cyber,rtp_alx_en,rtp_alx_zh,rtp_alx_es,rtp_alx_fr,rtp_alx_de,rtp_alx_ja,rtp_alx_ru,rtp_alx_hi" \
    --safe_dataset "helpful_base" \
    --max_safe_samples 25000 \
    --max_toxic_samples 2000 \
    --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
    --batch_size 64 \
    --seed 42

echo "LLM-based detectors focused experiment completed"
