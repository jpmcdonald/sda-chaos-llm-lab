# Experiments

Baseline vs experimental runs and reproduction scripts.

## Structure

- `baseline/` - Standard training runs with cross-entropy loss
- `uncertainty_aware/` - Experiments with SIP-like losses
- `regime_steering/` - Experiments with regime-aware policies
- `outer_loop/` - Experiments with PSO/SA/RL controllers
- `scripts/` - Scripts to reproduce metrics and plots

## Running Experiments

Each experiment should be reproducible and include:
- Configuration file
- Training script
- Evaluation script
- Results/logs directory

## Datasets

Experiments target:
- Small math/logic datasets (e.g., GSM8K-like subsets)
- Simple specification-generation tasks
- Chain-of-thought reasoning tasks

