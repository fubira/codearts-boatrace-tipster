"""Backtest boat 1 EV strategy over a date range with Kelly simulation.

Usage:
    uv run --directory ml python -m scripts.backtest_boat1 \
        --from 2026-03-19 --to 2026-04-03 \
        --bankroll 50000 --bet-cap 4000
"""

import argparse
import contextlib
import json
import sys

import numpy as np

from boatrace_tipster_ml.boat1_features import reshape_to_boat1
from boatrace_tipster_ml.boat1_model import train_boat1_model
from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df


def _load_stadium_names(db_path: str) -> dict[int, str]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT id, name FROM db.stadiums").fetchall()
        return {int(r[0]): r[1] for r in rows}
    finally:
        conn.close()


def backtest(
    from_date: str,
    to_date: str,
    db_path: str,
    bankroll: float,
    bet_cap: int,
    kelly_fraction: float,
    ev_threshold: float = 0.0,
) -> dict:
    with contextlib.redirect_stdout(sys.stderr):
        df = build_features_df(db_path)

    train_df = df[df["race_date"] < from_date]
    test_df = df[(df["race_date"] >= from_date) & (df["race_date"] < to_date)]

    if len(test_df) == 0:
        return {"error": f"No race data in {from_date} ~ {to_date}"}

    # Val = last ~2 months of train for early stopping
    dates = sorted(train_df["race_date"].unique())
    val_start = dates[max(0, len(dates) - 60)]

    with contextlib.redirect_stdout(sys.stderr):
        X_train, y_train, _ = reshape_to_boat1(train_df[train_df["race_date"] < val_start])
        X_val, y_val, _ = reshape_to_boat1(train_df[train_df["race_date"] >= val_start])
        X_test, y_test, meta_test = reshape_to_boat1(test_df)

        print(f"Train: {len(X_train)}R, Val: {len(X_val)}R, Test: {len(X_test)}R", file=sys.stderr)
        model, metrics = train_boat1_model(X_train, y_train, X_val, y_val)
        print(f"Val AUC: {metrics.get('val_auc', 'N/A')}", file=sys.stderr)

    probs = model.predict_proba(X_test)[:, 1]
    odds = meta_test["b1_tansho_odds"].values
    has_odds = ~np.isnan(odds)
    y_arr = y_test.values
    ev = np.where(has_odds, probs * odds - 1, np.nan)
    test_dates = meta_test["race_date"].values

    b1_test = test_df[test_df["boat_number"] == 1].reset_index(drop=True)
    stadium_names = _load_stadium_names(db_path)

    # --- EV summary (fixed ¥100 bet) ---
    ev_summary = []
    for ev_thr in [0, 10, 20, 30, 50]:
        mask = has_odds & (ev >= ev_thr / 100)
        n = int(mask.sum())
        if n == 0:
            continue
        wins = y_arr[mask]
        o = odds[mask]
        ev_summary.append({
            "ev_threshold": ev_thr,
            "bets": n,
            "hit_rate": round(float(wins.mean()), 4),
            "roi": round(float((wins * o).sum() / n), 4),
            "profit_at_100": round(((wins * o).sum() - n) * 100),
        })

    # --- Kelly simulation ---
    mask_ev0 = has_odds & (ev >= ev_threshold / 100)
    indices = np.where(mask_ev0)[0]

    bank = int(bankroll)
    daily: dict[str, dict] = {}
    n_bets = 0
    min_bank = bank
    max_bank = bank

    for idx in indices:
        date = str(test_dates[idx])
        p = float(probs[idx])
        o = float(odds[idx])
        won = bool(y_arr[idx])

        if p * o - 1 <= 0 or o <= 1:
            continue

        kelly_frac = min((p * o - 1) / (o - 1) * kelly_fraction, 0.05)
        bet = min(max(int(bank * kelly_frac / 100) * 100, 100), bet_cap)
        if bet > bank:
            bet = int(bank / 100) * 100
        if bet < 100:
            continue

        payout = round(bet * o) if won else 0
        bank += payout - bet
        min_bank = min(min_bank, bank)
        max_bank = max(max_bank, bank)
        n_bets += 1

        if date not in daily:
            daily[date] = {"bets": 0, "wagered": 0, "payout": 0, "wins": 0}
        daily[date]["bets"] += 1
        daily[date]["wagered"] += bet
        daily[date]["payout"] += payout
        if won:
            daily[date]["wins"] += 1

    # Build daily summary
    daily_summary = []
    cum_bank = int(bankroll)
    for d in sorted(daily.keys()):
        dl = daily[d]
        pl = dl["payout"] - dl["wagered"]
        cum_bank += pl
        daily_summary.append({
            "date": d,
            "bets": dl["bets"],
            "wins": dl["wins"],
            "wagered": dl["wagered"],
            "payout": dl["payout"],
            "pl": pl,
            "bankroll": cum_bank,
        })

    total_wagered = sum(d["wagered"] for d in daily.values())
    total_payout = sum(d["payout"] for d in daily.values())

    return {
        "from_date": from_date,
        "to_date": to_date,
        "n_races": len(y_arr),
        "boat1_win_rate": round(float(y_arr.mean()), 4),
        "val_auc": metrics.get("val_auc"),
        "ev_summary": ev_summary,
        "kelly": {
            "initial_bankroll": int(bankroll),
            "bet_cap": bet_cap,
            "kelly_fraction": kelly_fraction,
            "ev_threshold": ev_threshold,
            "final_bankroll": int(bank),
            "profit": int(bank - bankroll),
            "total_wagered": total_wagered,
            "total_payout": total_payout,
            "roi": round(total_payout / total_wagered, 4) if total_wagered > 0 else 0,
            "n_bets": n_bets,
            "min_bankroll": int(min_bank),
            "max_bankroll": int(max_bank),
        },
        "daily": daily_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest boat 1 EV strategy")
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--bankroll", type=float, default=50000)
    parser.add_argument("--bet-cap", type=int, default=4000)
    parser.add_argument("--kelly", type=float, default=0.25)
    parser.add_argument("--ev-threshold", type=float, default=0, help="EV threshold for Kelly bets (e.g. 20 for EV>=+20%%)")
    args = parser.parse_args()

    result = backtest(
        args.from_date, args.to_date, args.db_path,
        args.bankroll, args.bet_cap, args.kelly, args.ev_threshold,
    )
    json.dump(result, sys.stdout, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
