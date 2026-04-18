"""Tests for evaluate_period_sweep consistency with evaluate_period.

Two tiers:
  1. Pure helper tests (_make_purchase) — synthetic fixtures, always run.
  2. Integration tests — require p2_v3 model and real DB; verify that
     evaluate_period_sweep with a single ev level produces the same
     purchases as evaluate_period, and that a stricter ev threshold buys
     a subset of the races bought by a looser ev.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import load_model, load_model_meta
from scripts.analyze_model import (
    _RaceContext,
    _make_purchase,
    evaluate_period,
    evaluate_period_sweep,
)

# --- pure helper tests (synthetic, no DB) ---


def _mk_ctx(hit_combo: str = "1-2-3") -> _RaceContext:
    return _RaceContext(
        race_id=100, stadium_id=1, race_date="2026-04-01",
        top3_conc=0.7, gap23=0.15, b1_won=True, hit_combo=hit_combo,
        candidates=[],
    )


def test_make_purchase_hit():
    ctx = _mk_ctx(hit_combo="1-2-3")
    tickets = [("1-2-3", 8.0, 0.1), ("1-3-2", 12.0, -0.2)]
    p = _make_purchase(ctx, tickets)
    assert p.won is True
    assert p.cost == 200
    assert p.payout == 800.0
    assert p.b1_won is True
    assert p.tickets == tickets


def test_make_purchase_miss():
    ctx = _mk_ctx(hit_combo="1-4-5")
    tickets = [("1-2-3", 8.0, 0.1), ("1-3-2", 12.0, -0.2)]
    p = _make_purchase(ctx, tickets)
    assert p.won is False
    assert p.cost == 200
    assert p.payout == 0.0


def test_make_purchase_single_ticket_hit():
    ctx = _mk_ctx(hit_combo="1-3-2")
    tickets = [("1-3-2", 15.0, 0.05)]
    p = _make_purchase(ctx, tickets)
    assert p.won is True
    assert p.cost == 100
    assert p.payout == 1500.0


def test_make_purchase_copies_ctx_fields():
    ctx = _RaceContext(
        race_id=42, stadium_id=7, race_date="2026-03-15",
        top3_conc=0.82, gap23=0.21, b1_won=False, hit_combo="2-1-3",
        candidates=[],
    )
    p = _make_purchase(ctx, [])
    assert p.race_id == 42
    assert p.stadium_id == 7
    assert p.race_date == "2026-03-15"
    assert p.top3_conc == 0.82
    assert p.gap23 == 0.21
    assert p.b1_won is False
    assert p.hit_combo == "2-1-3"


# --- integration tests (require p2_v3 model + real DB) ---


@pytest.fixture(scope="module")
def _integration_setup():
    model_dir = (
        Path(__file__).resolve().parent.parent / "models" / "p2_v3" / "ranking"
    )
    if not (model_dir / "model.pkl").exists():
        pytest.skip(f"Model not available: {model_dir}")
    model = load_model(model_dir)
    meta = load_model_meta(model_dir)
    df = build_features_df(DEFAULT_DB_PATH)
    conn = get_connection(DEFAULT_DB_PATH)
    rows = conn.execute(
        "SELECT race_id, combination, odds FROM db.race_odds "
        "WHERE bet_type='3連単'"
    ).fetchall()
    conn.close()
    odds = {(int(r[0]), r[1]): float(r[2]) for r in rows}
    return model, meta, df, odds


def _purchase_key(p):
    # race_id is enough — evaluate_period emits at most one Purchase per race
    return (p.race_id, p.won, p.cost, p.payout)


def test_sweep_single_ev_matches_scalar(_integration_setup):
    """evaluate_period_sweep([ev_th]) の購入集合は evaluate_period と一致。"""
    model, meta, df, odds = _integration_setup
    ev_th = meta["strategy"].get("ev_threshold", 0.0)
    from_date, to_date = "2026-04-01", "2026-04-18"

    scalar_p, scalar_total = evaluate_period(
        model, meta, df, odds, from_date, to_date,
    )
    sweep_by_ev, sweep_total = evaluate_period_sweep(
        model, meta, df, odds, from_date, to_date, [ev_th],
    )

    assert scalar_total == sweep_total
    sweep_p = sweep_by_ev[ev_th]
    assert len(scalar_p) == len(sweep_p)
    assert sorted(_purchase_key(p) for p in scalar_p) == sorted(
        _purchase_key(p) for p in sweep_p
    )


def test_sweep_multi_ev_each_matches_scalar(_integration_setup):
    """複数 ev sweep の各 ev が、その ev 単独の evaluate_period と一致。"""
    model, meta, df, odds = _integration_setup
    # Copy meta to mutate ev_threshold per call without side effects.
    meta = dict(meta)
    meta["strategy"] = dict(meta.get("strategy", {}))
    ev_levels = [0.0, -0.25]
    from_date, to_date = "2026-04-01", "2026-04-18"

    sweep_by_ev, _ = evaluate_period_sweep(
        model, meta, df, odds, from_date, to_date, ev_levels,
    )

    for ev in ev_levels:
        meta["strategy"]["ev_threshold"] = ev
        scalar_p, _ = evaluate_period(
            model, meta, df, odds, from_date, to_date,
        )
        sweep_p = sweep_by_ev[ev]
        assert sorted(_purchase_key(p) for p in scalar_p) == sorted(
            _purchase_key(p) for p in sweep_p
        ), f"mismatch at ev={ev}"


def test_sweep_stricter_ev_buys_subset(_integration_setup):
    """ev=0 の購入レースは ev=-0.25 の購入レースの subset のはず。"""
    model, meta, df, odds = _integration_setup
    from_date, to_date = "2026-04-01", "2026-04-18"

    sweep_by_ev, _ = evaluate_period_sweep(
        model, meta, df, odds, from_date, to_date, [0.0, -0.25],
    )
    strict_races = {p.race_id for p in sweep_by_ev[0.0]}
    loose_races = {p.race_id for p in sweep_by_ev[-0.25]}
    assert strict_races.issubset(loose_races)
