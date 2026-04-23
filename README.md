# Trust The Typical

**ICLR 2026** · [OpenReview](https://openreview.net/forum?id=vfbeleLBWv)

Debargha Ganguly, Sreehari Sankar, Biyao Zhang, Vikash Singh, Kanan Gupta,
Harshini Kavuru, Alan Luo, Weicong Chen, Warren Morningstar, Raghu Machiraju,
Vipin Chaudhary

*Case Western Reserve University · University of Pittsburgh · The Ohio State University · Google Research*

---

Trust The Typical (T3) is a framework for LLM safety that treats guardrailing as
an out-of-distribution (OOD) detection problem. Instead of enumerating what is
harmful, T3 learns the distribution of acceptable prompts in a semantic space
and flags significant deviations. It requires no training on harmful examples,
yet achieves state-of-the-art performance across 18 benchmarks spanning
toxicity, hate speech, jailbreaking, multilingual harms, and over-refusal,
reducing false positive rates by up to 40× relative to specialized safety
models. A single model trained only on safe English text transfers to diverse
domains and 14+ languages without retraining, and a GPU-optimized integration
with vLLM enables continuous guardrailing during token generation with <6%
overhead on large workloads.

This repository contains the reference implementation and the code needed to
reproduce the experiments in the paper.

## Repository layout

```
api/            Core T3 (Forte) text OOD detector
baselines/      Reimplementations of the 20+ baselines compared in the paper
experiments/    Unified evaluation drivers (one per benchmark family)
jobs/           SLURM / shell scripts for the paper's large-scale sweeps
prompt_plus_analysis/
                LLM-augmented variant used in the OR-Bench study
vllm_integration/
                Online guardrailing hooks for vLLM
results/        Written at run time (CSV + JSON per run); tracked empty
```

## Installation

```bash
git clone https://github.com/DebarghaG/trust-the-typical.git
cd trust-the-typical
python -m venv env && source env/bin/activate
pip install -r requirements.txt
```

Python 3.12 and a CUDA-capable GPU are recommended. Sentence-transformer
models are downloaded from the Hugging Face Hub on first use.

### Credentials

Benchmarks that call external APIs expect the following environment variables.
No keys are hard-coded in this repository — set them in your shell before
running.

```bash
export HF_TOKEN=hf_...                 # Hugging Face (model downloads)
export OPENAI_API_KEY=sk-...           # OpenAI Omni Moderation baseline
export PERSPECTIVE_API_KEY=...         # Google Perspective baseline
```

## Quickstart

### Python

```python
from api.forte_text_api import ForteTextOODDetector

detector = ForteTextOODDetector(method="ocsvm", device="cuda:0")
detector.fit(safe_texts)                          # ID-only training

predictions = detector.predict(test_texts)        # +1 = safe, -1 = OOD
scores      = detector.predict_proba(test_texts)  # continuous OOD score
metrics     = detector.evaluate(id_texts, ood_texts)  # AUROC, FPR@95, AUPRC, F1
```

`method` accepts `"gmm"`, `"kde"`, or `"ocsvm"`. Default encoders are
`Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-m3`, and `intfloat/e5-large-v2`;
override via `model_names=[(short_name, hf_id), ...]`.

## Citation

```bibtex
@inproceedings{ganguly2026trust,
  title     = {Trust The Typical},
  author    = {Debargha Ganguly and Sreehari Sankar and Biyao Zhang and
               Vikash Singh and Kanan Gupta and Harshini Kavuru and Alan Luo and
               Weicong Chen and Warren Richard Morningstar and Raghu Machiraju and
               Vipin Chaudhary},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=vfbeleLBWv}
}
```

## License

A university invention disclosure has been filed with Case Western Reserve
University for this work. The code is free to use without restriction for
academic and non-commercial research. For commercial licensing please contact
Debargha Ganguly (<debargha@case.edu>) and CWRU's Office of Research and
Technology Management (https://research.case.edu/tto/). See [LICENSE](LICENSE)
for the full terms.

## Acknowledgments

This work was funded by NSF Awards 2117439 and 2112606, and builds on prior
work in representation typicality (Forte; Ganguly et al., 2025) and PRDC
metrics. We thank the authors of the datasets and baseline implementations we
compare against.
