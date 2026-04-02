"""Tests for depth.utils.config."""

import argparse
import pytest
from utils.config import (
    TrainingConfig,
    DecoderConfig,
    FusionConfig,
    TranslatorConfig,
    MAEConfig,
    add_common_args,
)


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.architecture == "jepa"
        assert cfg.epochs == 80
        assert cfg.batch_size == 32
        assert cfg.learning_rate == pytest.approx(3e-4)
        assert cfg.dataset == "easy"
        assert cfg.task == "presence"
        assert cfg.seed == 42
        assert cfg.snr_db is None

    def test_custom_values(self):
        cfg = TrainingConfig(architecture="lewm", epochs=50, dataset="extreme", task="counting")
        assert cfg.architecture == "lewm"
        assert cfg.epochs == 50
        assert cfg.dataset == "extreme"
        assert cfg.task == "counting"

    def test_early_stopping_defaults(self):
        cfg = TrainingConfig()
        assert cfg.early_stop_patience == 15
        assert cfg.early_stop_min_delta == pytest.approx(0.001)

    def test_snr_db_can_be_set(self):
        cfg = TrainingConfig(snr_db=20.0)
        assert cfg.snr_db == pytest.approx(20.0)

    def test_from_args_with_full_namespace(self):
        args = argparse.Namespace(
            command="lewm",
            model="conv",
            epochs=100,
            batch_size=64,
            lr=1e-3,
            weight_decay=0.01,
            dataset="hard",
            task="counting",
            n_chunks=10,
            with_aug=True,
            light_aug=False,
            rotation_degrees=45,
            label_smoothing=0.0,
            use_focal_loss=False,
            sigreg_weight=0.2,
            seed=99,
            patience=20,
            min_delta=0.005,
            snr_db=10.0,
        )
        cfg = TrainingConfig.from_args(args)
        assert cfg.architecture == "lewm"
        assert cfg.epochs == 100
        assert cfg.dataset == "hard"
        assert cfg.seed == 99
        assert cfg.snr_db == pytest.approx(10.0)

    def test_from_args_with_missing_attrs_uses_defaults(self):
        # Namespace with no attributes at all — all getattr calls fall back to defaults
        args = argparse.Namespace()
        cfg = TrainingConfig.from_args(args)
        assert cfg.architecture == "jepa"
        assert cfg.epochs == 80
        assert cfg.dataset == "easy"
        assert cfg.snr_db is None

    def test_from_args_snr_db_none(self):
        args = argparse.Namespace(snr_db=None)
        cfg = TrainingConfig.from_args(args)
        assert cfg.snr_db is None


class TestDecoderConfig:
    def test_defaults(self):
        cfg = DecoderConfig()
        assert cfg.epochs == 50
        assert cfg.batch_size == 16
        assert cfg.dataset == "easy"

    def test_custom(self):
        cfg = DecoderConfig(epochs=100, dataset="hard")
        assert cfg.epochs == 100
        assert cfg.dataset == "hard"


class TestFusionConfig:
    def test_defaults(self):
        cfg = FusionConfig()
        assert cfg.dropout_prob == pytest.approx(0.5)
        assert cfg.task == "presence"

    def test_counting_task(self):
        cfg = FusionConfig(task="counting")
        assert cfg.task == "counting"


class TestTranslatorConfig:
    def test_defaults(self):
        cfg = TranslatorConfig()
        assert cfg.d_model == 256
        assert cfg.patch_size == 16
        assert cfg.snr_db is None

    def test_snr_ablation(self):
        cfg = TranslatorConfig(snr_db=5.0)
        assert cfg.snr_db == pytest.approx(5.0)


class TestMAEConfig:
    def test_defaults(self):
        cfg = MAEConfig()
        assert cfg.mask_ratio == pytest.approx(0.75)
        assert cfg.epochs == 100


class TestAddCommonArgs:
    def test_adds_dataset_arg(self):
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args(["--dataset", "extreme"])
        assert args.dataset == "extreme"

    def test_adds_patience_arg(self):
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args(["--patience", "30"])
        assert args.patience == 30

    def test_adds_snr_db_arg(self):
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args(["--snr-db", "15.0"])
        assert args.snr_db == pytest.approx(15.0)

    def test_snr_db_defaults_to_none(self):
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args([])
        assert args.snr_db is None

    def test_with_aug_flag(self):
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args_off = parser.parse_args([])
        args_on = parser.parse_args(["--with-aug"])
        assert not args_off.with_aug
        assert args_on.with_aug

    def test_invalid_dataset_choice_raises(self):
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--dataset", "impossible"])
