"""Command-line interface for EduDataAnalyzer."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from .data_loader import ensure_minimum_rows, load_dataset
from .metrics import compute_summary_metrics, cohort_metrics
from .model import FEATURE_COLUMNS, predict_risk, train_risk_model
from .report import generate_report


def _print_json(data: Dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_summarize(args: argparse.Namespace) -> None:
    df = load_dataset(args.path)
    summary = compute_summary_metrics(df)
    cohorts = cohort_metrics(df, by=args.by)

    print("Summary metrics:")
    _print_json(summary)
    print("\nCohort breakdown:")
    print(cohorts.to_markdown(index=False))


def cmd_predict(args: argparse.Namespace) -> None:
    df = load_dataset(args.path)
    ensure_minimum_rows(df)
    trained = train_risk_model(df)

    record = {feature: getattr(args, feature) for feature in FEATURE_COLUMNS}
    score = predict_risk(trained.model, [record])[0]
    print(f"Predicted dropout probability: {score:.3f}")


def cmd_report(args: argparse.Namespace) -> None:
    output = generate_report(args.path, args.output)
    print(f"Report written to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze educational datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="Print dataset metrics")
    summarize.add_argument("path", help="Path to CSV dataset")
    summarize.add_argument(
        "--by",
        default="program",
        help="Column used to group cohorts (default: program)",
    )
    summarize.set_defaults(func=cmd_summarize)

    predict = subparsers.add_parser("predict", help="Predict dropout probability")
    predict.add_argument("path", help="Path to CSV dataset used for training")
    for feature in FEATURE_COLUMNS:
        predict.add_argument(
            f"--{feature}",
            type=float,
            required=True,
            help=f"Value for feature '{feature}'",
        )
    predict.set_defaults(func=cmd_predict)

    report = subparsers.add_parser("report", help="Generate Markdown report")
    report.add_argument("path", help="Path to CSV dataset")
    report.add_argument(
        "--output",
        default="report.md",
        help="Where to write the report (default: report.md)",
    )
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
