"""Tests for depth.core.factory — get_trainer routing."""

import pytest
import torch
from unittest.mock import patch, MagicMock
from core.factory import get_trainer
from utils.config import (
    TrainingConfig,
    DecoderConfig,
    FusionConfig,
    TranslatorConfig,
    MAEConfig,
)


@pytest.fixture(scope="module")
def cpu():
    return torch.device("cpu")


class TestGetTrainer:
    def test_lewm_config_returns_lewm_trainer(self, cpu):
        from core.lewm_trainer import LeWMTrainer
        cfg = TrainingConfig(architecture="lewm")
        trainer = get_trainer(cfg, cpu)
        assert isinstance(trainer, LeWMTrainer)

    def test_lewm_plus_config_returns_lewm_plus_trainer(self, cpu):
        from core.lewm_plus_trainer import LeWMPlusTrainer
        cfg = TrainingConfig(architecture="lewm_plus")
        trainer = get_trainer(cfg, cpu)
        assert isinstance(trainer, LeWMPlusTrainer)

    def test_jepa_config_returns_jepa_trainer(self, cpu):
        from core.jepa_trainer import JEPATrainer
        cfg = TrainingConfig(architecture="jepa")
        trainer = get_trainer(cfg, cpu)
        assert isinstance(trainer, JEPATrainer)

    def test_unknown_architecture_defaults_to_jepa(self, cpu):
        from core.jepa_trainer import JEPATrainer
        cfg = TrainingConfig(architecture="nonexistent_arch")
        trainer = get_trainer(cfg, cpu)
        assert isinstance(trainer, JEPATrainer)

    def test_decoder_config_returns_decoder_trainer(self, cpu):
        from core.trainers_advanced import DecoderTrainer
        # DecoderTrainer loads pre-trained JEPA weights on init; patch that out
        with patch.object(DecoderTrainer, '_load_jepa_model', return_value=MagicMock()):
            cfg = DecoderConfig()
            trainer = get_trainer(cfg, cpu)
        assert isinstance(trainer, DecoderTrainer)

    def test_fusion_config_returns_fusion_trainer(self, cpu):
        from core.trainers_advanced import FusionTrainer
        cfg = FusionConfig()
        trainer = get_trainer(cfg, cpu)
        assert isinstance(trainer, FusionTrainer)

    def test_translator_config_returns_translator_trainer(self, cpu):
        from core.trainers_advanced import TranslatorTrainer
        cfg = TranslatorConfig()
        trainer = get_trainer(cfg, cpu)
        assert isinstance(trainer, TranslatorTrainer)

    def test_mae_config_returns_mae_trainer(self, cpu):
        from core.trainers_advanced import MAETrainer
        cfg = MAEConfig()
        trainer = get_trainer(cfg, cpu)
        assert isinstance(trainer, MAETrainer)

    def test_trainer_has_model_attribute(self, cpu):
        cfg = TrainingConfig(architecture="lewm")
        trainer = get_trainer(cfg, cpu)
        assert hasattr(trainer, "model")

    def test_trainer_has_config_attribute(self, cpu):
        cfg = TrainingConfig(architecture="lewm")
        trainer = get_trainer(cfg, cpu)
        assert hasattr(trainer, "config")
        assert trainer.config is cfg
