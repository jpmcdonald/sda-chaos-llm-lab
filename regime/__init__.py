"""
Regime indicators and steering policies for training dynamics.

This module contains:
- Code to compute regime indicators (norms, volatility, chaos-inspired metrics)
- Simple rule-based "regime steering" policies
  - Adjust learning rate/clipping/data mix when thresholds are crossed
  - Monitor training dynamics as a dynamical system
"""

__version__ = "0.1.0"

