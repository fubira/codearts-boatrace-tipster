"""Tests for model registry helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boatrace_tipster_ml import registry


def test_to_prefix_boundary():
    assert registry._to_prefix(0) == "aa"
    assert registry._to_prefix(1) == "ab"
    assert registry._to_prefix(25) == "az"
    assert registry._to_prefix(26) == "ba"
    assert registry._to_prefix(675) == "zz"


def test_to_prefix_out_of_range():
    with pytest.raises(ValueError):
        registry._to_prefix(-1)
    with pytest.raises(ValueError):
        registry._to_prefix(26 * 26)


def test_get_active_model_dir(tmp_path, monkeypatch):
    active = tmp_path / "active.json"
    active.write_text(json.dumps({"model": "p2_v9"}))
    monkeypatch.setattr(registry, "ACTIVE_PATH", active)
    assert registry.get_active_model_name() == "p2_v9"
    assert registry.get_active_model_dir() == "models/p2_v9"


def test_get_active_model_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "ACTIVE_PATH", tmp_path / "missing.json")
    with pytest.raises(RuntimeError, match="Missing active model config"):
        registry.get_active_model_name()


def test_get_active_model_no_field(tmp_path, monkeypatch):
    active = tmp_path / "active.json"
    active.write_text(json.dumps({}))
    monkeypatch.setattr(registry, "ACTIVE_PATH", active)
    with pytest.raises(RuntimeError, match="missing 'model'"):
        registry.get_active_model_name()


def test_next_prefix_advances(tmp_path, monkeypatch):
    counter = tmp_path / ".run-counter"
    counter.write_text("5\n")
    monkeypatch.setattr(registry, "COUNTER_PATH", counter)
    assert registry.peek_prefix() == "af"
    assert registry.next_prefix() == "af"
    assert counter.read_text().strip() == "6"
    assert registry.next_prefix() == "ag"
    assert counter.read_text().strip() == "7"


def test_next_prefix_initial(tmp_path, monkeypatch):
    counter = tmp_path / ".run-counter"  # does not exist
    monkeypatch.setattr(registry, "COUNTER_PATH", counter)
    assert registry.peek_prefix() == "aa"
    assert registry.next_prefix() == "aa"
    assert counter.read_text().strip() == "1"
