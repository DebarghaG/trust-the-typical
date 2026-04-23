#!/bin/bash
# Master submission script for all toxicity detection experiments
# Usage: bash jobs/submit_all_experiments.sh [experiment_type]
# experiment_type can be: all, core, ablation, stress, llm

echo "Toxicity Detection Experiments - NeurIPS Paper"
echo "============================================="

# Check if logs directory exists
if [ ! -d "logs" ]; then
    echo "Creating logs directory..."
    mkdir -p logs
fi

# Function to submit a job and track its ID
submit_job() {
    local script_name=$1
    local job_description=$2

    echo "Submitting: $job_description"
    job_id=$(sbatch jobs/$script_name | awk '{print $4}')
    echo "  Job ID: $job_id"
    echo "  Script: jobs/$script_name"
    echo "  Logs: logs/${script_name%.*}_${job_id}.out"
    echo ""
    return $job_id
}

# Parse command line argument
EXPERIMENT_TYPE=${1:-"all"}

case $EXPERIMENT_TYPE in
    "core")
        echo "Submitting core experiments..."
        submit_job "large_scale_id_experiments.sh" "Large-scale in-distribution comprehensive evaluation"
        submit_job "comprehensive_ood_experiments.sh" "Comprehensive out-of-distribution evaluation"
        ;;

    "ablation")
        echo "Submitting ablation studies..."
        submit_job "multilingual_ablation_experiments.sh" "Multilingual performance ablation"
        submit_job "embedding_model_ablation.sh" "Embedding model ablation (E5)"
        submit_job "domain_specific_experiments.sh" "Domain-specific performance evaluation"
        submit_job "cross_lingual_generalization.sh" "Cross-lingual generalization study"
        ;;

    "stress")
        echo "Submitting stress tests..."
        submit_job "high_volume_stress_test.sh" "High-volume stress test (50K samples)"
        ;;

    "llm")
        echo "Submitting LLM-focused experiments..."
        submit_job "llm_based_detectors_focus.sh" "LLM-based detectors comprehensive evaluation"
        ;;

    "lightweight")
        echo "Submitting lightweight experiments..."
        submit_job "lightweight_baselines_comparison.sh" "Lightweight baselines comparison"
        ;;

    "all")
        echo "Submitting ALL experiments..."
        submit_job "large_scale_id_experiments.sh" "Large-scale in-distribution comprehensive evaluation"
        submit_job "comprehensive_ood_experiments.sh" "Comprehensive out-of-distribution evaluation"
        submit_job "multilingual_ablation_experiments.sh" "Multilingual performance ablation"
        submit_job "embedding_model_ablation.sh" "Embedding model ablation (E5)"
        submit_job "domain_specific_experiments.sh" "Domain-specific performance evaluation"
        submit_job "cross_lingual_generalization.sh" "Cross-lingual generalization study"
        submit_job "high_volume_stress_test.sh" "High-volume stress test (50K samples)"
        submit_job "llm_based_detectors_focus.sh" "LLM-based detectors comprehensive evaluation"
        submit_job "lightweight_baselines_comparison.sh" "Lightweight baselines comparison"
        ;;

    *)
        echo "Usage: bash jobs/submit_all_experiments.sh [experiment_type]"
        echo ""
        echo "Available experiment types:"
        echo "  all        - Submit all experiments (default)"
        echo "  core       - Core ID/OOD experiments"
        echo "  ablation   - Ablation studies (multilingual, embedding, domain)"
        echo "  stress     - High-volume stress tests"
        echo "  llm        - LLM-based detector focus"
        echo "  lightweight- Non-LLM baselines only"
        echo ""
        echo "Examples:"
        echo "  bash jobs/submit_all_experiments.sh core"
        echo "  bash jobs/submit_all_experiments.sh ablation"
        echo "  bash jobs/submit_all_experiments.sh all"
        exit 1
        ;;
esac

echo "Submission complete!"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Cancel all jobs with: scancel -u $USER"
echo "View logs in: logs/"
echo ""
