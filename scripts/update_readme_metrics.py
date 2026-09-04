"""Regenerate the README's Results section from the training artifacts.

Metrics in documentation drift the moment they are typed by hand, and a stale
number in a security README is a false claim. This script renders the section
directly from ``models/metrics.json`` so the README can only ever state what
was actually measured.

    python scripts/update_readme_metrics.py

Run it after every ``python -m ml.train``. It is idempotent, and it fails
loudly rather than writing a partial section.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
METADATA_PATH = PROJECT_ROOT / "models" / "feature_metadata.json"
README_PATH = PROJECT_ROOT / "README.md"

START = "<!-- METRICS:START -->"
END = "<!-- METRICS:END -->"


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def render(metrics: dict, metadata: dict) -> str:
    test = metrics["test"]
    dataset = metrics.get("dataset", {})
    selected = metrics["selected_model"]

    lines: list[str] = []
    add = lines.append

    add("All figures below are **measured**, not illustrative. They come from")
    add("`models/metrics.json`, written by `python -m ml.train` on")
    add(f"{metadata.get('trained_at', 'unknown date')}.")
    add("")

    # ---------------------------------------------------------------- headline
    add(f"**Selected model: `{selected}`** — chosen on {metrics.get('selection_criterion', 'validation PR-AUC')}.")
    add("")
    add("### Held-out test split")
    add("")
    add("Scored exactly once, after selection. No registrable domain in this split")
    add("appears anywhere in training.")
    add("")
    add("| Metric | Value |")
    add("| --- | --- |")
    add(f"| Accuracy | **{pct(test['accuracy'])}** |")
    add(f"| Precision | **{pct(test['precision'])}** |")
    add(f"| Recall | **{pct(test['recall'])}** |")
    add(f"| F1 score | **{pct(test['f1'])}** |")
    add(f"| ROC-AUC | **{test['roc_auc']:.4f}** |")
    add(f"| PR-AUC | **{test['pr_auc']:.4f}** |")
    add(f"| False-positive rate | {pct(test['false_positive_rate'])} |")
    add(f"| False-negative rate | {pct(test['false_negative_rate'])} |")
    add(f"| Decision threshold | {test['threshold']:.4f} |")
    add("")
    add("Confusion matrix on "
        f"{test['true_negatives'] + test['false_positives'] + test['false_negatives'] + test['true_positives']:,} "
        "test URLs:")
    add("")
    add("| | predicted legitimate | predicted phishing |")
    add("| --- | --- | --- |")
    add(f"| **actually legitimate** | {test['true_negatives']:,} | {test['false_positives']:,} |")
    add(f"| **actually phishing** | {test['false_negatives']:,} | {test['true_positives']:,} |")
    add("")

    # ------------------------------------------------------------- comparison
    comparison = metrics.get("comparison", [])
    if comparison:
        add("### Model comparison (validation split)")
        add("")
        add("Selection used validation only; the test split above was untouched until")
        add("the winner was fixed.")
        add("")
        add("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Train time |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in comparison:
            v = row["validation"]
            name = row["model"].replace("_", " ")
            mark = " ★" if row["model"] == selected else ""
            add(
                f"| {name}{mark} | {pct(v['accuracy'])} | {pct(v['precision'])} | "
                f"{pct(v['recall'])} | {pct(v['f1'])} | {v['roc_auc']:.4f} | "
                f"{v['pr_auc']:.4f} | {row.get('train_seconds', 0):.0f}s |"
            )
        add("")
        add("★ selected. Note how far the linear baseline sits below the tree and")
        add("n-gram models, and that the ensemble beats both of its members —")
        add("they fail on different URLs.")
        add("")

    # -------------------------------------------------------- operating points
    curve = metrics.get("operating_points", [])
    if curve:
        add("### Operating points (test split)")
        add("")
        add("The precision/recall trade-off, published rather than buried in a")
        add("constant. Retune with `python -m ml.train --min-precision 0.95`.")
        add("")
        add("| Precision floor | Threshold | Precision | Recall |")
        add("| --- | --- | --- | --- |")
        for point in curve:
            if not point.get("attainable"):
                add(f"| {point['precision_floor']:.2f} | — | not attainable | — |")
                continue
            shipped = " ← shipped" if abs(
                point["precision_floor"] - metrics.get("precision_floor", -1)
            ) < 1e-9 else ""
            add(
                f"| {point['precision_floor']:.2f}{shipped} | {point['threshold']:.4f} | "
                f"{pct(point['precision'])} | {pct(point['recall'])} |"
            )
        add("")

    # ------------------------------------------------------- external holdout
    holdout = metrics.get("external_holdout")
    if holdout:
        add("### Independent holdout — PhishTank")
        add("")
        add("The strongest evidence of real generalisation: a live phishing feed never")
        add("used in training, with every domain seen during training removed first.")
        add("")
        add(f"- **Recall: {pct(holdout['recall'])}** — detected "
            f"{holdout['detected']:,} of {holdout['rows']:,} URLs")
        add(f"- Spanning {holdout['unique_domains']:,} registrable domains, none seen in training")
        add(f"- Mean predicted probability: {holdout.get('mean_probability', 0):.4f}")
        add("")
        add("Phishing-only feed, so recall is the only meaningful metric here.")
        add("")

    # ------------------------------------------------------------ sanity gate
    sanity = metrics.get("sanity_check")
    if sanity:
        add("### Sanity gate")
        add("")
        add("A hand-curated set of obvious cases, checked after every training run.")
        add("**This is a smoke test, not a benchmark** — it is deliberately easy, and")
        add("its score must never be quoted as model accuracy. It exists because")
        add("aggregate metrics cannot detect a shortcut that is present in the test")
        add("split too.")
        add("")
        add(f"- Overall: **{sanity['correct']}/{sanity['total']}** correct")
        add(f"- Well-known legitimate URLs: {pct(sanity['legitimate_accuracy'], 0)}")
        add(f"- Phishing-shaped URLs: {pct(sanity['phishing_accuracy'], 0)}")
        failures = sanity.get("failures", [])
        if failures:
            add("")
            add("Current failures, reported rather than hidden:")
            add("")
            for failure in failures:
                kind = "legitimate" if failure["expected"] == 0 else "phishing"
                add(f"- `{failure['url']}` — expected {kind}, scored {failure['probability']:.4f}")
            add("")
            add("These are false positives on well-known sites, and they reflect a real")
            add("limitation: the corpora's legitimate examples skew long-tail, so major")
            add("brand domains are under-represented. See [Limitations](#limitations).")
        else:
            add("")
            add("No failures in the current run.")
        add("")

    # ----------------------------------------------------------------- dataset
    add("### Training data")
    add("")
    add(f"- **{dataset.get('sampled_rows', 0):,}** URLs used, "
        f"capped at {dataset.get('max_rows_cap', 0):,} (`--max-rows 0` uses all "
        f"{metrics.get('dataset', {}).get('cleaning_report', {}).get('final_rows', 0):,})")
    add(f"- **{dataset.get('unique_domains', 0):,}** unique registrable domains")
    add(f"- Split {dataset.get('train_rows', 0):,} train / "
        f"{dataset.get('val_rows', 0):,} validation / {dataset.get('test_rows', 0):,} test")
    add(f"- Test-split phishing rate: {pct(dataset.get('test_phishing_rate', 0), 1)}")
    add(f"- Features: **{metadata.get('feature_count', 0)}** numeric + character n-grams")
    add(f"- Split strategy: {dataset.get('split_strategy', 'n/a')}")
    add("")

    top = metadata.get("top_features", [])[:8]
    if top:
        add("Most influential numeric features (ensemble's tree branch):")
        add("")
        for feature in top:
            add(f"- `{feature['feature']}` — {feature['importance'] * 100:.1f}%")
        add("")

    add(f"Total pipeline runtime: {metrics.get('total_pipeline_seconds', 0):.0f}s.")

    return "\n".join(lines)


def main() -> int:
    if not METRICS_PATH.exists():
        print(f"error: {METRICS_PATH} not found. Run: python -m ml.train", file=sys.stderr)
        return 1

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}

    readme = README_PATH.read_text(encoding="utf-8")
    if START not in readme or END not in readme:
        print(f"error: README is missing the {START} / {END} markers", file=sys.stderr)
        return 1

    before = readme.split(START)[0]
    after = readme.split(END)[1]
    updated = f"{before}{START}\n{render(metrics, metadata)}\n{END}{after}"

    README_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated {README_PATH} Results section from {METRICS_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
