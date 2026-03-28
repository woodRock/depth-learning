"""Factory function for creating trainers."""

from typing import TYPE_CHECKING, Any
import torch

if TYPE_CHECKING:
    from .base import BaseTrainer


def get_trainer(config: Any, device: torch.device) -> "BaseTrainer":
    """
    Factory function to get appropriate trainer for model type.
    
    Args:
        config: Training configuration (TrainingConfig, DecoderConfig, etc.)
        device: Device to train on (cuda, mps, cpu)
    
    Returns:
        Appropriate trainer instance
    """
    from .jepa_trainer import JEPATrainer
    from .lewm_trainer import LeWMTrainer
    from .lewm_plus_trainer import LeWMPlusTrainer
    from .trainers_advanced import DecoderTrainer, FusionTrainer, TranslatorTrainer, MAETrainer
    from utils.config import DecoderConfig, FusionConfig, TranslatorConfig, MAEConfig
    
    logger = __import__('utils.logging', fromlist=['get_logger']).get_logger(__name__)
    
    # Determine model type from config
    model_type = getattr(config, 'model_type', None)
    architecture = getattr(config, 'architecture', 'jepa')
    
    if isinstance(config, DecoderConfig):
        architecture = "decoder"
    elif isinstance(config, FusionConfig):
        architecture = "fusion"
    elif isinstance(config, TranslatorConfig):
        architecture = "translator"
    elif isinstance(config, MAEConfig):
        architecture = "mae"
        
    logger.info(f"Creating trainer for architecture: {architecture} (model_type: {model_type})")
    
    if architecture == "lewm":
        return LeWMTrainer(config, device)
    elif architecture == "lewm_plus":
        return LeWMPlusTrainer(config, device)
    elif architecture == "decoder":
        return DecoderTrainer(config, device)
    elif architecture == "fusion":
        return FusionTrainer(config, device)
    elif architecture == "translator":
        return TranslatorTrainer(config, device)
    elif architecture == "mae":
        return MAETrainer(config, device)
    else:
        # Default to JEPA trainer (transformer, conv, lstm, ast)
        return JEPATrainer(config, device)
