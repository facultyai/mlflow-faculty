from datetime import datetime

import pytest
from pytz import UTC

from mlflow_faculty.py23 import to_timestamp


@pytest.mark.parametrize(
    "dt, expected_timestamp",
    [
        (datetime(1970, 1, 1, tzinfo=UTC), 0),
        (datetime(2019, 3, 6, 14, 57, 51, tzinfo=UTC), 1551884271),
        (datetime(2019, 3, 6, 14, 57, 51, 987000, tzinfo=UTC), 1551884271.987),
    ],
)
def test_to_timestamp(dt, expected_timestamp):
    assert to_timestamp(dt) == expected_timestamp
