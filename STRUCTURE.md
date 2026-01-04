# Repository Structure

This document outlines the directory structure of the sda-chaos-llm-lab repository.

```
sda-chaos-llm-lab/
├── README.md                 # Main project documentation
├── LICENSE                   # License file
├── STRUCTURE.md              # This file
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Python package configuration
├── .gitignore                # Git ignore rules
│
├── configs/                  # Training configurations
│   ├── README.md
│   └── example_config.yaml   # Example configuration template
│
├── models/                   # Model architectures
│   └── __init__.py
│
├── losses/                   # Loss functions
│   └── __init__.py
│   # Planned: cross_entropy.py, uncertainty_aware.py, sip_like.py
│
├── regime/                   # Regime indicators and steering
│   └── __init__.py
│   # Planned: indicators.py, steering.py, metrics.py
│
├── outer_loop/               # Outer-loop optimization
│   └── __init__.py
│   # Planned: pso.py, simulated_annealing.py, rl_controller.py
│
├── utils/                    # Utility functions
│   └── __init__.py
│   # Planned: data_utils.py, training_utils.py, viz_utils.py
│
├── data/                     # Data storage and processing
│   ├── README.md
│   ├── raw/                  # Original datasets (gitignored)
│   ├── processed/            # Preprocessed datasets (gitignored)
│   └── scripts/              # Data preprocessing scripts
│
├── experiments/              # Experiment runs and scripts
│   ├── README.md
│   ├── scripts/              # Training and evaluation scripts
│   │   └── README.md
│   ├── baseline/             # Baseline experiment results
│   ├── uncertainty_aware/    # Uncertainty-aware loss experiments
│   ├── regime_steering/      # Regime steering experiments
│   └── outer_loop/           # Outer-loop optimization experiments
│
└── notebooks/                # Jupyter notebooks for analysis
    └── README.md
    # Planned: trajectory_analysis.ipynb, regime_visualization.ipynb, etc.
```

## Next Steps

1. **Models**: Implement small transformer architectures or wrappers
2. **Losses**: Implement standard and uncertainty-aware loss functions
3. **Regime**: Implement regime indicators and steering policies
4. **Outer Loop**: Implement PSO/SA/RL controllers
5. **Experiments**: Create training and evaluation scripts
6. **Data**: Add data preprocessing scripts and download utilities

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Development

This is a research sandbox. Start with small experiments in:
- `experiments/baseline/` for standard training
- `notebooks/` for exploratory analysis
- Individual modules for focused development

