# ace-step-sae: Sparse Autoencoder Interpretability for ACE-Step
#
# Implementation of "Discovering and Steering Interpretable Concepts
# in Large Generative Music Models" (Singh, Cherep & Maes, 2025)
# adapted for ACE-Step's Diffusion Transformer architecture.

from ace_step_sae.config import SAEConfig
from ace_step_sae.model import TopKSparseAutoencoder
from ace_step_sae.hooks import ActivationCollector
from ace_step_sae.trainer import SAETrainer
from ace_step_sae.steering import SteeringHook
from ace_step_sae.features import FeatureAnalyzer

__version__ = "0.1.0"

__all__ = [
    "SAEConfig",
    "TopKSparseAutoencoder",
    "ActivationCollector",
    "SAETrainer",
    "SteeringHook",
    "FeatureAnalyzer",
]
