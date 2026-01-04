# Configs

Training configurations for baseline vs experimental runs.

## Structure

- Model configurations (architecture, size, initialization)
- Dataset configurations (data sources, splits, preprocessing)
- Hyperparameter configurations (learning rates, batch sizes, etc.)
- Policy configurations (regime steering rules, outer-loop settings)

## Format

Configs can be in YAML, JSON, or Python format. Consider using:
- YAML for human-readable configs
- JSON for programmatic configs
- Python for complex configs with logic

## Example

```yaml
model:
  name: "small-transformer"
  num_layers: 6
  hidden_size: 512
  num_heads: 8

training:
  learning_rate: 1e-4
  batch_size: 32
  num_epochs: 10

regime:
  indicators: ["gradient_norm", "loss_volatility"]
  thresholds:
    gradient_norm: 1.0
    loss_volatility: 0.1
```

