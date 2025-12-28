from pathlib import Path

from edudataanalyzer.data_loader import load_dataset
from edudataanalyzer.metrics import compute_summary_metrics, cohort_metrics


def test_summary_metrics_shape():
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_students.csv"
    df = load_dataset(data_path)

    summary = compute_summary_metrics(df)

    assert summary["count"] == len(df)
    assert 0 <= summary["pass_rate"] <= 1
    assert 0 <= summary["avg_attendance"] <= 1
    assert "grade_attendance_corr" in summary


def test_cohort_metrics_rows():
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_students.csv"
    df = load_dataset(data_path)

    cohorts = cohort_metrics(df, by="program")
    # Ensure every program is represented once
    assert set(cohorts["program"]) == set(df["program"].unique())
