"""
Loss functions for LLM training.

This module contains:
- Standard cross-entropy baselines
- SIP-like / uncertainty-aware variants:
  - Soft token sets
  - Interval-aware numeric loss
  - Sample weighting based on uncertainty
"""

__version__ = "0.1.0"

