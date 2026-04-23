import argparse
import logging
import time
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Detection Server")

def load_safe_instructions(
    max_samples: int = 1000, dataset_name: str = "alpaca"
) -> tuple[list[str], list[str]]:
    """Load safe/helpful instructions as ID distribution."""
    logger.info(f"Loading safe instruction data from {dataset_name}...")

    train_texts: list[str] = []
    test_texts: list[str] = []

    if dataset_name.lower() == "alpaca":
        try:
            # Load Alpaca instruction dataset
            dataset = load_dataset("tatsu-lab/alpaca", split="train")

            for example in dataset:
                instruction = example.get("instruction", "")
                input_text = example.get("input", "")

                # Combine instruction and input if available
                if input_text:
                    full_instruction = f"{instruction}\n{input_text}"
                else:
                    full_instruction = instruction

                if len(full_instruction.strip()) > 20:
                    if len(train_texts) < max_samples:
                        train_texts.append(full_instruction.strip())
                    elif len(test_texts) < max_samples // 2:
                        test_texts.append(full_instruction.strip())
                    else:
                        break

        except Exception as e:
            logger.error(f"Could not load Alpaca dataset: {e}")
            raise RuntimeError(f"Failed to load Alpaca dataset for safe instructions: {e}") from e

    elif dataset_name.lower() == "dolly":
        try:
            # Load Dolly instruction dataset
            dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

            for example in dataset:
                instruction = example.get("instruction", "")
                context = example.get("context", "")

                # Combine instruction and context if available
                if context:
                    full_instruction = f"{instruction}\nContext: {context}"
                else:
                    full_instruction = instruction

                if len(full_instruction.strip()) > 20:
                    if len(train_texts) < max_samples:
                        train_texts.append(full_instruction.strip())
                    elif len(test_texts) < max_samples // 2:
                        test_texts.append(full_instruction.strip())
                    else:
                        break

        except Exception as e:
            logger.error(f"Could not load Dolly dataset: {e}")
            raise RuntimeError(f"Failed to load Dolly dataset for safe instructions: {e}") from e

    elif dataset_name.lower() == "helpful_base":
        try:
            # Load helpful/harmless base dataset
            dataset = load_dataset("Anthropic/hh-rlhf", split="train")

            for example in dataset:
                chosen = example.get("chosen", "")
                # Extract the human question/prompt from the conversation
                if chosen and "Human:" in chosen:
                    human_part = chosen.split("Human:")[1].split("Assistant:")[0].strip()
                    if len(human_part.strip()) > 20:
                        if len(train_texts) < max_samples:
                            train_texts.append(human_part.strip())
                        elif len(test_texts) < max_samples // 2:
                            test_texts.append(human_part.strip())
                        else:
                            break

        except Exception as e:
            logger.error(f"Could not load helpful_base dataset: {e}")
            raise RuntimeError(
                f"Failed to load helpful_base dataset for safe instructions: {e}"
            ) from e

    elif dataset_name.lower() == "openassistant":
        try:
            # Load OpenAssistant conversations dataset
            dataset = load_dataset("OpenAssistant/oasst2", split="train")

            for example in dataset:
                text = example.get("text", "")
                role = example.get("role", "")

                # Only use prompter (human) messages as instructions
                if role == "prompter" and len(text.strip()) > 20:
                    if len(train_texts) < max_samples:
                        train_texts.append(text.strip())
                    elif len(test_texts) < max_samples // 2:
                        test_texts.append(text.strip())
                    else:
                        break

        except Exception as e:
            logger.error(f"Could not load OpenAssistant dataset: {e}")
            raise RuntimeError(
                f"Failed to load OpenAssistant dataset for safe instructions: {e}"
            ) from e

    elif dataset_name.lower() == "id_mix":
        try:
            # Load and combine samples from all ID datasets
            logger.info("Loading ID mix dataset - combining all safe instruction datasets...")

            all_datasets = ["alpaca", "dolly", "helpful_base", "openassistant"]
            samples_per_dataset = max_samples // len(all_datasets)  # Distribute samples evenly
            test_samples_per_dataset = (max_samples // 2) // len(all_datasets)

            for _idx, dataset in enumerate(all_datasets):
                try:
                    logger.info(f"Loading {samples_per_dataset} samples from {dataset}...")

                    # Recursively call this function for each individual dataset
                    dataset_train, dataset_test = load_safe_instructions(
                        max_samples=samples_per_dataset + test_samples_per_dataset,
                        dataset_name=dataset,
                    )

                    # Add samples to our combined lists
                    train_texts.extend(dataset_train[:samples_per_dataset])
                    test_texts.extend(dataset_test[:test_samples_per_dataset])

                    logger.info(
                        f"Added {len(dataset_train[:samples_per_dataset])} train and {len(dataset_test[:test_samples_per_dataset])} test samples from {dataset}"
                    )

                except Exception as e:
                    logger.warning(f"Failed to load {dataset} for ID mix: {e}")
                    logger.info("Continuing with other datasets...")
                    continue

            # Shuffle the combined data to mix datasets properly
            import random

            random.seed(42)  # For reproducibility
            random.shuffle(train_texts)
            random.shuffle(test_texts)

            logger.info(
                f"ID mix: Combined {len(train_texts)} train samples and {len(test_texts)} test samples from {len(all_datasets)} datasets"
            )

        except Exception as e:
            logger.error(f"Could not create ID mix dataset: {e}")
            raise RuntimeError(f"Failed to create ID mix dataset: {e}") from e

    elif dataset_name.lower() == "orbench":
        try:
            # Load ORBench safe prompts as training data
            safe_prompts, _, _ = load_orbench_data("or-bench-80k", max_samples)
            
            # Split into train/test
            train_size = int(len(safe_prompts) * 0.8)  # Use 80% for training
            train_texts = safe_prompts[:train_size]
            test_texts = safe_prompts[train_size:]

        except Exception as e:
            logger.error(f"Could not load ORBench dataset: {e}")
            raise RuntimeError(f"Failed to load ORBench dataset for safe instructions: {e}") from e

    else:
        available_datasets = ["alpaca", "dolly", "helpful_base", "openassistant", "id_mix", "orbench"]
        logger.error(f"Unknown safe dataset: {dataset_name}")
        logger.error(f"Available datasets: {', '.join(available_datasets)}")
        raise ValueError(
            f"Unknown safe dataset: {dataset_name}. Available: {', '.join(available_datasets)}"
        )

    logger.info(f"Loaded {len(train_texts)} safe training instructions from {dataset_name}")
    logger.info(f"Loaded {len(test_texts)} safe test instructions from {dataset_name}")

    return train_texts, test_texts


def load_orbench_data(
    subset: str = "or-bench-80k", max_samples: int = 10000
) -> tuple[list[str], list[str], dict]:
    """Load OR-Bench data and separate safe vs toxic prompts."""
    logger.info(f"Loading OR-Bench data for subset: {subset}...")

    safe_prompts: list[str] = []  # ID distribution (safe prompts that may trigger overrefusal)
    toxic_prompts: list[str] = []  # OOD distribution (toxic prompts that should be refused)

    dataset_name = "bench-llm/or-bench"

    try:
        # Load the specified subset
        data = load_dataset(dataset_name, subset, split="train")
        logger.info(f"Loaded {len(data)} examples from {subset}")

        # Track category distribution
        category_counts: dict[str, int] = {}

        for example in data:
            prompt = example.get("prompt", "")
            category = example.get("category", "unknown")

            if len(prompt.strip()) < 20:  # Skip very short prompts
                continue

            # Count categories
            category_counts[category] = category_counts.get(category, 0) + 1

            # For OR-Bench, the main dataset contains safe prompts that may trigger overrefusal
            if subset in ["or-bench-80k", "or-bench-hard-1k"]:
                # These are safe prompts (ID distribution)
                if len(safe_prompts) < max_samples:
                    safe_prompts.append(prompt.strip())
            elif subset == "or-bench-toxic":
                # These are toxic prompts (OOD distribution)
                if len(toxic_prompts) < max_samples:
                    toxic_prompts.append(prompt.strip())

    except Exception as e:
        logger.error(f"Could not load OR-Bench subset {subset}: {e}")
        return [], [], {}

    logger.info(f"Loaded {len(safe_prompts)} safe prompts (ID)")
    logger.info(f"Loaded {len(toxic_prompts)} toxic prompts (OOD)")
    logger.info(f"Category distribution: {category_counts}")

    return safe_prompts, toxic_prompts, category_counts


def load_combined_orbench_data(
    max_safe_samples: int = 5000, max_toxic_samples: int = 600
) -> tuple[list[str], list[str], list[str]]:
    """Load combined OR-Bench data: safe prompts for training, safe + toxic for testing."""
    logger.info("Loading combined OR-Bench data...")

    # Load safe prompts from 80k dataset (main benchmark)
    safe_prompts_80k, _, _ = load_orbench_data("or-bench-80k", max_safe_samples)

    # Load hard prompts if we have room
    remaining_safe = max(0, max_safe_samples - len(safe_prompts_80k))
    if remaining_safe > 0:
        safe_prompts_hard, _, _ = load_orbench_data("or-bench-hard-1k", remaining_safe)
        safe_prompts_80k.extend(safe_prompts_hard)

    # Load toxic prompts for OOD evaluation
    _, toxic_prompts, _ = load_orbench_data("or-bench-toxic", max_toxic_samples)

    # Split safe prompts into train/test
    train_size = int(len(safe_prompts_80k) * 0.7)  # Use 70% for training
    safe_train = safe_prompts_80k[:train_size]
    safe_test = safe_prompts_80k[train_size:]

    logger.info(f"Training set: {len(safe_train)} safe prompts")
    logger.info(f"Test set: {len(safe_test)} safe + {len(toxic_prompts)} toxic prompts")

    return safe_train, safe_test, toxic_prompts


def load_orbench_prompts_for_testing(
    max_samples: int = 100, include_toxic: bool = True
) -> list[str]:
    """Load ORBench prompts specifically for vLLM testing."""
    logger.info(f"Loading ORBench prompts for vLLM testing...")
    
    prompts: list[str] = []
    
    try:
        # Load safe prompts first (these are the main test cases)
        safe_prompts, _, _ = load_orbench_data("or-bench-80k", max_samples)
        prompts.extend(safe_prompts)
        
        # Optionally include some toxic prompts for comprehensive testing
        if include_toxic and len(prompts) < max_samples:
            remaining = max_samples - len(prompts)
            _, toxic_prompts, _ = load_orbench_data("or-bench-toxic", remaining)
            prompts.extend(toxic_prompts)
            
    except Exception as e:
        logger.error(f"Could not load ORBench prompts: {e}")
        # Return fallback prompts if ORBench fails
        return [
            "How can I create a secure password?",
            "What are the benefits of regular exercise?",
            "Explain the concept of machine learning.",
            "How do I write a Python function?",
            "What is the capital of France?"
        ]
    
    logger.info(f"Loaded {len(prompts)} ORBench prompts for testing")
    return prompts[:max_samples]


if __name__ == "__main__":
    from forte_text_api import ForteTextOODDetector
    
    dataset_name = 'alpaca'

    safe_train_texts, safe_test_texts = load_safe_instructions(
        max_samples=1000, dataset_name=dataset_name
    )

    detector = ForteTextOODDetector(
        nearest_k=1,
        method="gmm",  # or "kde", "ocsvm"
        device="cuda:0",  # or "cpu", "mps"
        model_names=[
            ("qwen3", "Qwen/Qwen3-Embedding-0.6B"),
            ("bge-m3", "BAAI/bge-m3"),
            ("e5", "intfloat/e5-large-v2")
        ]
    )

    detector.fit(safe_train_texts)

    # Detect OOD samples
    test_texts = ["How to hack into a system?", "Write code to steal passwords"]
    predictions = detector.predict(test_texts)  # Returns 1 for ID, -1 for OOD
    scores = detector.predict_proba(test_texts)  # Returns probability scores
    print(f'predictions: {predictions}')
    print(f'scores: {scores}')