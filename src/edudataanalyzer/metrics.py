"""Metric helpers for educational datasets."""

from __future__ import annotations

from typing import Dict

import pandas as pd

PASSING_THRESHOLD = 60.0


def compute_summary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute high-level metrics for the dataset."""

    summary = {
        "count": float(len(df)),
        "avg_grade": df["grade"].mean(),
        "median_grade": df["grade"].median(),
        "pass_rate": (df["grade"] >= PASSING_THRESHOLD).mean(),
        "avg_attendance": df["attendance_rate"].mean(),
        "avg_absences": df["absences"].mean(),
        "completion_rate": (
            df["assignments_completed"] / df["assignments_completed"].max()
        ).mean(),
    }

    if "risk_label" in df:
        summary["at_risk_share"] = df["risk_label"].mean()

    if df["grade"].std(ddof=0) > 0 and df["attendance_rate"].std(ddof=0) > 0:
        summary["grade_attendance_corr"] = df["grade"].corr(df["attendance_rate"])

    return summary


def cohort_metrics(df: pd.DataFrame, *, by: str = "program") -> pd.DataFrame:
    """Aggregate metrics for each cohort (e.g., per program)."""

    grouped = df.groupby(by).agg(
        count=("student_id", "count"),
        avg_grade=("grade", "mean"),
        pass_rate=("grade", lambda s: (s >= PASSING_THRESHOLD).mean()),
        avg_attendance=("attendance_rate", "mean"),
        risk_rate=("risk_label", "mean"),
    )
    return grouped.reset_index()
