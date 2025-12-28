"""Utilities for loading and validating educational datasets."""

from __future__ import annotations

import pathlib
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = {
    "student_id",
    "program",
    "grade",
    "attendance_rate",
    "assignments_completed",
    "absences",
    "risk_label",
}


def _validate_columns(columns: Iterable[str]) -> None:
    missing = REQUIRED_COLUMNS.difference(set(columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def load_dataset(path: str | pathlib.Path, *, dropna: bool = True) -> pd.DataFrame:
    """Load the student dataset and enforce the expected schema.

    Parameters
    ----------
    path: str | pathlib.Path
        Path to the CSV file.
    dropna: bool
        Whether to drop rows with missing values in required columns.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for downstream analysis.
    """

    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    _validate_columns(df.columns)

    if dropna:
        df = df.dropna(subset=REQUIRED_COLUMNS)

    # Ensure dtypes are sensible for numeric operations
    numeric_cols = [
        "grade",
        "attendance_rate",
        "assignments_completed",
        "absences",
        "risk_label",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if dropna:
        df = df.dropna(subset=numeric_cols)

    # Keep identifiers as strings to avoid accidental numeric formatting issues
    df["student_id"] = df["student_id"].astype(str)
    df["program"] = df["program"].astype(str)

    return df.reset_index(drop=True)


def ensure_minimum_rows(df: pd.DataFrame, *, minimum: int = 20) -> None:
    """Guardrail to avoid training on tiny datasets."""

    if len(df) < minimum:
        raise ValueError(
            f"Dataset too small for robust analysis (found {len(df)} rows, need {minimum}+)."
        )
