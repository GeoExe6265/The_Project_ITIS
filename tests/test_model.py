from pathlib import Path

import numpy as np

from edudataanalyzer.data_loader import load_dataset
from edudataanalyzer.model import FEATURE_COLUMNS, predict_risk, train_risk_model


def test_model_trains_and_predicts():
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_students.csv"
    df = load_dataset(data_path)

    trained = train_risk_model(df, test_size=0.25, random_state=0)
    assert "accuracy" in trained.metrics

    sample_record = {feature: df.iloc[0][feature] for feature in FEATURE_COLUMNS}
    score = predict_risk(trained.model, [sample_record])
    assert isinstance(score, np.ndarray)
    assert score.shape == (1,)
    assert 0 <= score[0] <= 1
