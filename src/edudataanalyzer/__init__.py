"""EduDataAnalyzer package for analyzing educational datasets."""

__all__ = [
    "load_dataset",
    "compute_summary_metrics",
    "train_risk_model",
    "predict_risk",
    "generate_report",
]

from .data_loader import load_dataset
from .metrics import compute_summary_metrics
from .model import predict_risk, train_risk_model
from .report import generate_report

__version__ = "0.1.0"
