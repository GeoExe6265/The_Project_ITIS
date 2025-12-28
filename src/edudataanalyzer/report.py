"""Generate lightweight Markdown reports from datasets."""

from __future__ import annotations

import pathlib
from datetime import datetime

from .data_loader import ensure_minimum_rows, load_dataset
from .metrics import compute_summary_metrics, cohort_metrics
from .model import FEATURE_COLUMNS, predict_risk, train_risk_model


def _format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def generate_report(
    dataset_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    *,
    top_n: int = 5,
) -> pathlib.Path:
    """Produce a Markdown report summarizing the dataset and model performance."""

    df = load_dataset(dataset_path)
    ensure_minimum_rows(df, minimum=20)

    summary = compute_summary_metrics(df)
    cohorts = cohort_metrics(df)
    trained = train_risk_model(df)

    # Identify highest-risk students for monitoring
    risk_scores = predict_risk(trained.model, df[FEATURE_COLUMNS].to_dict(orient="records"))
    df_with_scores = df.copy()
    df_with_scores["risk_score"] = risk_scores
    top_students = df_with_scores.sort_values("risk_score", ascending=False).head(top_n)

    lines: list[str] = []
    lines.append("# EduDataAnalyzer Report\n")
    lines.append(f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z\n")
    lines.append("## Dataset Summary")
    lines.append(f"- Records: {int(summary['count'])}")
    lines.append(f"- Average grade: {summary['avg_grade']:.2f}")
    lines.append(f"- Pass rate: {_format_percentage(summary['pass_rate'])}")
    lines.append(f"- Attendance: {summary['avg_attendance']:.2f}")
    if "at_risk_share" in summary:
        lines.append(f"- At-risk share: {_format_percentage(summary['at_risk_share'])}")
    if "grade_attendance_corr" in summary:
        lines.append(
            f"- Grade/attendance correlation: {summary['grade_attendance_corr']:.2f}"
        )
    lines.append("")

    lines.append("## Cohort Breakdown (by program)")
    lines.append(cohorts.to_markdown(index=False))
    lines.append("")

    lines.append("## Model Performance (logistic regression)")
    for metric, value in trained.metrics.items():
        if metric == "roc_auc":
            lines.append(f"- {metric}: {value:.3f}")
        else:
            lines.append(f"- {metric}: {value:.3f}")
    lines.append("")

    lines.append(f"## Top {top_n} Students by Risk Score")
    lines.append("student_id | program | grade | attendance | risk_score")
    lines.append(":--|:--|--:|--:|--:")
    for _, row in top_students.iterrows():
        lines.append(
            f"{row['student_id']} | {row['program']} | {row['grade']:.1f} | {row['attendance_rate']:.2f} | {row['risk_score']:.2f}"
        )
    lines.append("")

    output_path = pathlib.Path(output_path)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
