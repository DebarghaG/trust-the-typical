# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment (adjust path as needed)
# source /users/PAS2271/debargha/distribution-guardrails/env/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Large-scale OR-Bench overrefusal detection experiment with ALL baselines
echo "Starting large-scale OR-Bench overrefusal detection experiment"
# echo "Job ID: $SLURM_JOB_ID"
# echo "Node: $SLURMD_NODENAME"
# echo "GPU: $CUDA_VISIBLE_DEVICES"

python unified_baselines_orbench_overrefusal_detection.py \
    --methods "Forte_GMM,Forte_OCSVM" \
    --max_safe_samples 40000 \
    --max_toxic_samples 10000 \
    --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
    --analysis_model_name "facebook/opt-125m" \
    --batch_size 32 \
    --seed 42

echo "Large-scale OR-Bench overrefusal experiment completed"


# python unified_baselines_orbench_overrefusal_detection.py --methods "Forte_GMM,Forte_OCSVM" --max_safe_samples 40000 --max_toxic_samples 10000 --embedding_model "Qwen/Qwen3-Embedding-0.6B" --batch_size 32 --seed 42