Below is a first-pass **design/specification document** you can drop into `docs/specs/00_overview.md` (or `DESIGN.md`) and iterate.

***

# sda-chaos-llm-lab – Experiment Architecture Specification

## Overview

This document specifies the architecture and experimental design for the **sda-chaos-llm-lab**, a small‑scale research environment for applying **Sequential Decision Analytics (SDA)**, **chaos/regime control**, and **uncertainty-aware objectives** to the training and steering of small language models (LLMs).[1][2][3][4]

The lab is built around three core ideas:

1. **Training as a sequential decision problem**  
2. **Training dynamics as regimes in a dynamical system**  
3. **Loss functions and data weighting as uncertainty/SIP controls**[5][6][7]

This document defines the main components, interfaces, and initial experiments.

***

## System goals and constraints

### Goals

- Provide a **reproducible experimental harness** for:
  - SDA‑style training policies for small LMs.  
  - Regime/chaos‑aware monitoring and interventions.  
  - Uncertainty‑aware (SIP‑like) objectives and sample weighting.  
- Run on **commodity hardware** (e.g., Mac M2 Max, 64 GB RAM) using:
  - Models up to ~1–2B parameters for routine experiments.  
  - Short‑horizon training runs that can be repeated many times.  
- Enable **paired comparisons**:
  - Baseline vs experimental training policies.  
  - Plain CE vs uncertainty-aware loss.  
  - Static vs regime-aware schedules.

### Non-goals (initial phase)

- Frontier-scale pretraining.  
- Production-ready serving infrastructure.  
- Highly optimized CUDA/custom-kernel work.

***

## High-level architecture

The lab is organized into the following logical layers:

1. **Core model layer** – small transformer models or wrappers over existing small LMs.  
2. **Data & task layer** – small, focused benchmarks (math, logic, spec generation) with clear metrics.[8][9][10]
3. **Training engine** – standard gradient-based training with pluggable loss, scheduler, and logging.  
4. **Regime monitoring & control layer** – computes regime indicators and applies interventions.  
5. **Outer-loop policy / meta-optimization layer** – chooses training decisions (hyperparameters, curricula) over episodes.  
6. **Experiment orchestration layer** – defines experiments, runs sweeps, and collects results.

The interaction pattern is:

> Outer-loop policy → config for training episode → training engine + regime controller → metrics → outer-loop update.

***

## 1. Core model layer

### Models

Initial target:

- **Size**: 100–500M parameters (custom) and 1–2B (pretrained) for most experiments.  
- **Type**: standard decoder-only transformer with:
  - Multi-head self-attention.  
  - Feed-forward MLP blocks.  
  - LayerNorm / RMSNorm, residual connections.[11]

### Interface

Abstract `Model` interface:

```python
class Model(nn.Module):
    def forward(self, input_ids, attention_mask=None):
        # returns logits: [batch, seq_len, vocab_size]
        ...
```

Requirements:

- Compatible with token-level cross-entropy and soft-target distributions.  
- Supports saving / loading checkpoints.  
- For pretrained models: thin wrapper around HF/MLX models.

***

## 2. Data & task layer

### Task types

Initial task families:

1. **Math/logic QA**  
   - Small GSM8K-like subsets or synthetic math word problems.[9][10][8]
   - Metrics: exact match (EM), accuracy, step-by-step reasoning quality (optional).

2. **Short spec generation / structured text**  
   - Toy “requirements → spec” pairs.  
   - Metrics: sequence-level accuracy, structured extraction quality.

3. **Calibration tasks**  
   - Simple classification / multiple choice tasks where **calibration** can be measured:
     - Reliability diagrams, ECE, Brier score.[7]

### Data interface

Abstract `TaskDataset`:

```python
class TaskDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        # returns input_ids, labels, optional metadata (e.g., uncertainty tags)
        ...
```

Datasets should support:

- Train / validation split.  
- Optional per-sample **uncertainty metadata**:
  - Flags for “noisy” / “ambiguous” samples.  
  - Target token sets (for SIP-like labeling).

***

## 3. Training engine

### Core training loop

A generic trainer:

```python
class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, regime_controller, logger, config):
        ...

    def train(self, train_loader, val_loader, episode_config) -> Dict[str, Any]:
        # runs one training episode (N steps/epochs) under episode_config
        # returns metrics: final val loss, accuracy, calibration, stability stats, logs
        ...
```

Key pluggable components:

- **Loss function**: cross-entropy vs SIP/uncertainty-aware variants.  
- **Optimizer / scheduler**: AdamW, Adafactor, etc.; LR schedule defined by episode config.  
- **Regime controller**: optional; receives live stats and can adjust LR, clipping, curriculum.  

### Episode configuration

`episode_config` encodes the **training decisions** for one run:

- Learning rate schedule parameters (initial LR, warmup steps, decay).  
- Gradient clipping threshold.  
- Weight decay / regularization coeffs.  
- Proportion of “hard” vs “easy” samples per step (data mix).  
- Loss variant and its hyperparameters (e.g., label smoothing factor, SIP weights).  
- Regime controller on/off and its thresholds.

This is the object manipulated by the **outer-loop policy**.

***

## 4. Regime monitoring & control layer

### Regime indicators

At each training step or block of steps, compute a `RegimeState`:

```python
@dataclass
class RegimeState:
    step: int
    train_loss: float
    loss_ema: float
    loss_volatility: float
    grad_norm: float
    grad_norm_ema: float
    param_update_norm: float
    activation_stats: Dict[str, float]  # e.g., layerwise means/stds
    anomaly_flags: Dict[str, bool]
```

Example indicators:

- **Loss dynamics**:
  - Instant loss, EMA, and short/medium-horizon volatility.  
- **Gradient metrics**:
  - Global gradient norm, EMA; fraction of steps above a threshold.  
- **Parameter dynamics**:
  - Norm of parameter updates relative to parameter norm.  
- **Activation statistics**:
  - Layer-wise mean/std; out-of-range detection.

These are **proxies** for regime shifts, not full chaos analysis, but they capture transitions between:

- Stable learning.  
- Overfitting / flattening.  
- Divergence / instability.  

### Regime controller

A `RegimeController` maps `RegimeState` → interventions:

```python
class RegimeController:
    def update(self, state: RegimeState, episode_config: EpisodeConfig) -> EpisodeConfig:
        # returns possibly modified episode_config (or runtime decisions)
        ...
```

Initial rules (simple, interpretable):

- If `grad_norm` exceeds threshold X for Y consecutive steps:
  - Reduce LR by factor α.  
  - Increase gradient clipping.  

- If loss volatility spikes beyond threshold:
  - Switch to easier data subset (curriculum back-off).  

- If loss stagnates and metrics stagnate:
  - Slight LR increase or curriculum shift to harder samples.

More advanced controllers can be plugged in later (e.g., learned policies).

***

## 5. Loss functions: baseline vs SIP-like

### Baseline: standard cross-entropy

Token-level cross-entropy with one-hot targets:

```python
loss = cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
```

### SIP / uncertainty-aware variants

Initial designs:

1. **Soft token sets**  
   - For some tasks (e.g., numeric responses), define a **soft target distribution** over tokens:
     - Main correct token gets high probability.  
     - Nearby or equivalent tokens get lower but non-zero mass.[12]
   - Implemented via KL divergence between predicted distribution and target distribution.

2. **Per-sample uncertainty weighting**  
   - Each training example gets an uncertainty weight \(w_i\):  
     - High uncertainty → lower weight; low uncertainty → higher weight.  
   - Loss: \( L = \sum_i w_i \cdot \text{CE}(p_\theta, y_i) \).[5][7]

3. **Hybrid calibration term**  
   - Add a calibration-oriented regularizer (e.g., entropy or ECE proxy) to encourage well-calibrated probabilities.[7]

The loss module interface:

```python
class LossModule:
    def __call__(self, logits, labels, batch_metadata) -> Dict[str, torch.Tensor]:
        # returns {"loss": loss, "aux": {...}}
        ...
```

***

## 6. Outer-loop policy / meta-optimization layer

### Episode-level optimization

The outer loop treats each training run as an **episode**:

- Given prior results, choose an `episode_config`.  
- Run `Trainer.train` for a fixed budget (e.g., N steps or epochs).  
- Observe summary metrics (fitness).  
- Update the policy / search algorithm.

### Policy/search methods (initial)

1. **Random/grid search baseline**  
   - Baseline for comparison: simple random or grid search over LR, clipping, etc.

2. **Particle Swarm Optimization (PSO)**  
   - Represent each particle as a vector of hyperparameters (LR, clipping, curriculum mix, SIP weights).  
   - Fitness: validation metric after fixed steps (e.g., accuracy + stability penalty).[13][14]

3. **Simulated Annealing (SA)**  
   - Start from a baseline config; propose local perturbations; accept/reject based on fitness with temperature schedule.[15][16]

Later extensions can include RL-based policies, but initial focus is on PSO/SA as global search methods.

### Interface

```python
class OuterLoopOptimizer:
    def suggest(self) -> EpisodeConfig:
        ...

    def observe(self, episode_config: EpisodeConfig, metrics: Dict[str, Any]):
        ...
```

The experiment runner wires:

- `OuterLoopOptimizer` → `EpisodeConfig` → `Trainer` → metrics → `OuterLoopOptimizer.observe`.

***

## 7. Experiment orchestration

### Experiment types

Initial experimental templates:

1. **Loss comparison experiment**  
   - Fixed architecture, optimizer, and schedule.  
   - Compare:
     - CE-only vs SIP/uncertainty-aware loss.  
   - Metrics:
     - Val loss, EM/accuracy.  
     - Calibration (ECE, Brier score).  
     - Stability indicators (divergence, variance across seeds).[7]

2. **Regime controller experiment**  
   - Fixed base hyperparameters.  
   - Compare:
     - No regime controller vs controller with simple rules.  
   - Metrics:
     - Frequency of instability events.  
     - Performance variance across runs.  
     - Final task performance.

3. **Outer-loop optimization experiment**  
   - Compare:
     - Random search vs PSO vs SA on the same hyperparameter space.  
   - Metrics:
     - Best-found validation performance per unit compute.  
     - Robustness of found configs (re-run best configs multiple times).

### Orchestration interface

A simple CLI / script:

```bash
python -m experiments.run \
  --experiment loss_comparison_gsm8k \
  --config configs/loss_comparison_gsm8k.yaml
```

Each experiment script should:

- Load base config.  
- Initialize outer-loop optimizer (or fixed configs).  
- Run N episodes.  
- Save:
  - Configs, metrics, logs, and selected training curves.  

***

## 8. Logging, metrics, and artifacts

### Logging

- Step-level logs:
  - Loss, regime indicators, key metrics.  
- Episode-level summary:
  - Final val metrics, stability stats, regime transitions, runtime.  

Use JSON/CSV + optional tensorboard/MLflow for easy analysis.

### Artifacts

- Model checkpoints (for post-hoc evaluation).  
- Plots:
  - Loss vs steps.  
  - Regime indicators over time.  
  - Calibration curves.  
  - Outer-loop optimization trajectories.

***

## 9. Roadmap (initial phases)

**Phase 1 – Baseline & scaffolding**

- Implement:
  - Small model + basic dataset loaders.  
  - Baseline trainer with CE.  
  - Logging + simple experiment script.

**Phase 2 – SIP / uncertainty-aware loss**

- Add:
  - Soft target distributions for numeric/structured tokens.  
  - Uncertainty-based sample weighting.  
- Run:
  - Loss comparison experiments on a small math/logic task.

**Phase 3 – Regime monitoring & simple controller**

- Implement regime state and indicators.  
- Add rule-based controller (LR/clipping/curriculum adjustments).  
- Run:
  - Regime controller vs none experiments.

**Phase 4 – Outer-loop PSO/SA**

- Implement PSO and SA over episode configs.  
- Run:
  - PSO/SA vs random search experiments.

Each phase should produce at least one **notebook and write-up** documenting findings and guiding future extensions.

***

This specification is intended as a living document. As you implement components and run experiments, you can refine interfaces, add new regime indicators, and extend the outer-loop to more sophisticated policies.

[1](https://connect.informs.org/blogs/warren-powell/2023/06/01/SDAM)
[2](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119815068)
[3](https://tamids.tamu.edu/wp-content/uploads/2022/01/Slides-Warren-Powell-Jan-24-2022.pdf)
[4](https://arxiv.org/html/2404.05782v1)
[5](https://aclanthology.org/2024.acl-long.597.pdf)
[6](https://machinelearningmastery.com/prediction-intervals-for-machine-learning/)
[7](https://www.sciencedirect.com/science/article/pii/S2666827025001562)
[8](https://huggingface.co/papers/2312.09241)
[9](https://www.emergentmind.com/papers/2312.09241)
[10](https://arxiv.org/pdf/2312.09241.pdf)
[11](https://en.wikipedia.org/wiki/Neural_network_(machine_learning))
[12](https://arxiv.org/html/2411.02083v3)
[13](https://airccse.org/journal/ijcsea/papers/1211ijcsea06.pdf)
[14](https://www.sciencedirect.com/science/article/pii/S2773186323000312)
[15](https://pmc.ncbi.nlm.nih.gov/articles/PMC8947290/)
[16](https://aicompetence.org/particle-swarm-optimization-vs-genetic-algorithm/)