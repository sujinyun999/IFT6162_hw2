"""
Flash Calciner Control Package

This package provides:
- Physics-based simulation of a flash clay calciner
- Neural surrogate model for fast dynamics evaluation
- RL environments for both simplified (Part 1) and full (Part 2) problems
"""

from .physics import SimplifiedFlashCalciner, N_SPECIES, L
from .surrogate import (
    CalcinerSimulator,
    SpatiallyAwareDynamics,
    SurrogateModel,
    TransitionDataset,
    generate_training_data,
)
from .mpc import CalcinerDynamics
from .baselines import CalcinerEnv, ConstantTemperatureController, evaluate_baseline, evaluate_baseline_part2
from .surrogate_env import SurrogateCalcinerEnv

__all__ = [
    # Physics
    'SimplifiedFlashCalciner',
    'N_SPECIES',
    'L',
    # Surrogate
    'CalcinerSimulator',
    'SpatiallyAwareDynamics', 
    'SurrogateModel',
    'TransitionDataset',
    'generate_training_data',
    # Simplified dynamics
    'CalcinerDynamics',
    # RL Environments
    'CalcinerEnv',  # Part 1: simplified 3D state
    'SurrogateCalcinerEnv',  # Part 2: full 140D state
    'ConstantTemperatureController',
    'evaluate_baseline',
]
