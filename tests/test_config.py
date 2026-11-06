"""Tests for the config.

A silently-ignored typo in a config is worse than a crash: the run succeeds and reports a number
nobody asked for. So unknown keys are rejected, impossible fractions are rejected, and the
shipped YAML is checked against the dataclass it is supposed to populate.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import yaml

from fraud_audit.config import DEFAULT_CONFIG_PATH, AuditConfig, Tolerances
from fraud_audit.models import MODEL_NAMES
from fraud_audit.recalibrate import RECALIBRATORS


class TestShippedYaml:
    def test_exists_and_loads(self):
        assert DEFAULT_CONFIG_PATH.exists(), DEFAULT_CONFIG_PATH
        assert isinstance(AuditConfig.from_yaml(), AuditConfig)

    def test_every_key_is_a_real_field(self):
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        assert set(raw) <= set(AuditConfig.__dataclass_fields__)
        assert set(raw["tolerances"]) <= set(Tolerances.__dataclass_fields__)

    def test_names_a_model_that_exists(self):
        cfg = AuditConfig.from_yaml()
        assert cfg.subject in MODEL_NAMES

    def test_names_a_recalibrator_that_exists(self):
        cfg = AuditConfig.from_yaml()
        assert cfg.primary_recalibrator in {cls.name for cls in RECALIBRATORS}


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(train_frac=0.0),
            dict(train_frac=1.0),
            dict(recal_frac=-0.1),
            dict(intime_val_frac=1.5),
            dict(train_frac=0.9, recal_frac=0.2),
            dict(n_boot=0),
            dict(review_cost=0.0),
            dict(review_cost_grid=()),
        ],
    )
    def test_impossible_settings_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            AuditConfig(**kwargs)

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValueError, match="unknown configuration keys"):
            AuditConfig.from_mapping({"seed": 1, "trian_frac": 0.5})

    def test_unknown_tolerance_key_is_rejected(self):
        with pytest.raises(ValueError, match="unknown tolerance keys"):
            AuditConfig.from_mapping({"tolerances": {"auc_drop": 0.02, "sloop": [0.8, 1.2]}})

    def test_inverted_tolerance_band_is_rejected(self):
        with pytest.raises(ValueError, match="lo < hi"):
            Tolerances(oe=(1.25, 0.80))


class TestOverrides:
    def test_none_valued_overrides_are_ignored(self):
        """Lets a CLI pass every flag through without knowing which the user actually set."""
        cfg = AuditConfig(seed=7)
        assert cfg.with_overrides(seed=None, review_cost=None).seed == 7

    def test_supplied_overrides_win(self):
        cfg = AuditConfig.from_yaml(review_cost=12.5, seed=1)
        assert cfg.review_cost == 12.5
        assert cfg.seed == 1

    def test_overriding_an_unknown_key_is_rejected(self):
        with pytest.raises(ValueError, match="unknown configuration keys"):
            AuditConfig().with_overrides(nonsense=1)

    def test_config_is_immutable(self):
        """A run's parameters must not change halfway through it."""
        with pytest.raises(FrozenInstanceError):
            AuditConfig().seed = 99  # type: ignore[misc]

    def test_yaml_lists_become_tuples_so_the_config_stays_hashable_in_spirit(self):
        cfg = AuditConfig.from_yaml()
        assert isinstance(cfg.review_cost_grid, tuple)
        assert isinstance(cfg.tolerances.oe, tuple)


class TestSerialisation:
    def test_round_trips_through_a_mapping(self):
        cfg = AuditConfig.from_yaml()
        assert AuditConfig.from_mapping(cfg.to_dict()) == cfg

    def test_to_dict_is_json_friendly(self):
        import json

        json.dumps(AuditConfig.from_yaml().to_dict())
