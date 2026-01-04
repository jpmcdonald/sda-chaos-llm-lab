# Experiment Scripts

Scripts for running and reproducing experiments.

## Structure

- `train_baseline.py` - Standard training with cross-entropy
- `train_uncertainty_aware.py` - Training with SIP-like losses
- `train_regime_steering.py` - Training with regime-aware policies
- `train_outer_loop.py` - Training with outer-loop optimization
- `evaluate.py` - Model evaluation and metrics
- `plot_results.py` - Generate comparison plots and visualizations

## Usage

```bash
# Run a baseline experiment
python experiments/scripts/train_baseline.py --config configs/baseline_config.yaml

# Run with regime steering
python experiments/scripts/train_regime_steering.py --config configs/regime_config.yaml
```

