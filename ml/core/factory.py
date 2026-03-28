"""Factory function for creating trainers."""

from typing import TYPE_CHECKING
import torch

from utils.config import TrainingConfig

if TYPE_CHECKING:
    from .base import BaseTrainer


def get_trainer(config: TrainingConfig, device: torch.device) -> "BaseTrainer":
    """
    Factory function to get appropriate trainer for model type.
    
    Args:
        config: Training configuration
        device: Device to train on (cuda, mps, cpu)
    
    Returns:
        Appropriate trainer instance
    
    Raises:
        ValueError: If model_type is not recognized
    """
    from .jepa_trainer import JEPATrainer
    from .lewm_trainer import LeWMTrainer
    from .lewm_plus_trainer import LeWMPlusTrainer
    
    logger = __import__('utils.logging', fromlist=['get_logger']).get_logger(__name__)
    logger.info(f"Creating trainer for model type: {config.model_type}")
    
    if config.model_type == "lewm":
        logger.info("Using LeWM trainer (acoustic-only)")
        return LeWMTrainer(config, device)
    elif config.model_type == "lewm_plus":
        logger.info("Using LeWM++ trainer (multi-modal + SigReg)")
        return LeWMPlusTrainer(config, device)
    else:
        # Default to JEPA trainer for transformer, conv, lstm, ast
        logger.info(f"Using JEPA trainer ({config.model_type})")
        return JEPATrainer(config, device)
