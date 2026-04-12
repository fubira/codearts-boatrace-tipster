"""Promote a draft model to production.

Copies model files (model.pkl, model_meta.json) from draft to production directory.
Requires explicit confirmation to prevent accidental overwrites.

Usage:
    uv run --directory ml python -m scripts.promote_model
    uv run --directory ml python -m scripts.promote_model --draft models/draft --prod models/p2_v2
    uv run --directory ml python -m scripts.promote_model --component ranking
    uv run --directory ml python -m scripts.promote_model --yes  # skip confirmation
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from boatrace_tipster_ml.registry import get_active_model_dir

COMPONENTS = ["ranking", "boat1"]


def show_meta(model_dir: Path, label: str) -> bool:
    """Print model_meta.json summary. Returns True if model exists."""
    meta_path = model_dir / "model_meta.json"
    pkl_path = model_dir / "model.pkl"
    if not meta_path.exists() and not pkl_path.exists():
        print(f"  {label}: (not found)")
        return False

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        hp = meta.get("hyperparameters", {})
        st = meta.get("strategy", {})
        tr = meta.get("training", {})
        n_feat = len(meta.get("feature_columns", []))
        print(f"  {label}:")
        print(f"    {n_feat} features, n_est={hp.get('n_estimators', '?')}, lr={hp.get('learning_rate', '?')}")
        if st:
            print(f"    strategy: b1<{st.get('b1_threshold', '?')}, ev>{st.get('ev_threshold', '?')}")
        if tr:
            note = tr.get("note") or tr.get("date_range", "")
            print(f"    training: {note}")
    else:
        print(f"  {label}: model.pkl only (no meta)")
    return True


def promote(draft_dir: Path, prod_dir: Path, component: str) -> None:
    """Copy model files from draft to production."""
    src = draft_dir / component
    dst = prod_dir / component

    files = ["model.pkl", "model_meta.json"]
    if not all((src / f).exists() for f in files):
        missing = [f for f in files if not (src / f).exists()]
        print(f"  {component}: incomplete — missing {', '.join(missing)}")
        return

    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(src / f, dst / f)
    print(f"  {component}: {len(files)} file(s) copied → {dst}")


def main():
    parser = argparse.ArgumentParser(description="Promote draft model to production")
    parser.add_argument("--draft", default="models/draft", help="Draft model directory")
    parser.add_argument("--prod", default=get_active_model_dir(), help="Production model directory")
    parser.add_argument("--component", choices=COMPONENTS, default=None,
                        help="Promote only this component (default: all)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    draft = Path(args.draft)
    prod = Path(args.prod)
    components = [args.component] if args.component else COMPONENTS

    print(f"Draft:      {draft}")
    print(f"Production: {prod}")
    print()

    # Show current state
    has_draft = False
    for comp in components:
        print(f"[{comp}]")
        d = show_meta(draft / comp, "draft")
        show_meta(prod / comp, "prod")
        has_draft = has_draft or d
        print()

    if not has_draft:
        print("ERROR: No draft model found. Train first with train_ranking.py / train_boat1_binary.py")
        sys.exit(1)

    # Confirm
    if not args.yes:
        answer = input("Promote draft → production? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    # Promote
    print("Promoting...")
    for comp in components:
        if (draft / comp).exists():
            promote(draft, prod, comp)

    print("\nDone. Verify with: uv run python -m scripts.backtest_trifecta --from 2026-04-01 --to 2026-04-10")


if __name__ == "__main__":
    main()
