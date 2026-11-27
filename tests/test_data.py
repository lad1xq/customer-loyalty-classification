import numpy as np
import pandas as pd

from loyalty_segmentation.data import prepare_data


def test_preprocessing_handles_missing_and_categorical_values():
    frame = pd.DataFrame(
        {
            "customer_id": range(1, 13),
            "age": [20, 21, np.nan, 25, 30, 31, 32, 35, 40, 41, 42, 45],
            "region": ["Urban", "Rural", None] * 4,
            "customer_segment": ["Loyal", "Regular", "Occasional"] * 4,
        }
    )

    prepared = prepare_data(frame, test_size=0.25)

    assert prepared.x_train.shape[0] == 9
    assert prepared.x_test.shape[0] == 3
    assert np.isfinite(prepared.x_train).all()
    assert np.isfinite(prepared.x_test).all()

