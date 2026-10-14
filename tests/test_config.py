from pathlib import Path

import pytest

from wrestleai.config import Config, config_from_dict, load_config, parse_overrides

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_dataclass_defaults() -> None:
    assert load_config(ROOT / "configs" / "default.yaml") == Config()


def test_overrides_are_parsed_as_yaml_scalars() -> None:
    assert parse_overrides(["a.b=3", "a.c=false", "d=[1, 2]", "e=null"]) == {
        "a": {"b": 3, "c": False}, "d": [1, 2], "e": None,
    }


def test_round_trip_through_dict() -> None:
    cfg = load_config(ROOT / "configs" / "test.yaml", ["training.epochs=7"])
    assert config_from_dict(cfg.to_dict()) == cfg


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(None, ["training.epoch=3"])


def test_multiclass_labels_supported() -> None:
    cfg = load_config(None, ["labels.classes=[none, takedown, pin]", "labels.positive_class=takedown"])
    assert cfg.labels.num_classes == 3 and cfg.labels.default_class == "none"


def test_positive_class_must_exist() -> None:
    with pytest.raises(ValueError):
        load_config(None, ["labels.positive_class=nope"])
