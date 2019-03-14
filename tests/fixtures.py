from datetime import datetime

from mlflow.entities import Metric, Param, RunTag
from faculty.clients.experiment import (
    Metric as FacultyMetric,
    Param as FacultyParam,
    Tag as FacultyTag,
)

from pytz import UTC


METRIC_TIMESTAMP = datetime(2019, 3, 13, 17, 0, 15, 110000, tzinfo=UTC)
FACULTY_METRIC = FacultyMetric(
    key="metric-key", value="metric-value", timestamp=METRIC_TIMESTAMP
)
MLFLOW_METRIC = Metric(
    "metric-key", "metric-value", int(METRIC_TIMESTAMP.timestamp())
)

FACULTY_PARAM = FacultyParam(key="param-key", value="param-value")
MLFLOW_PARAM = Param("param-key", "param-value")

FACULTY_TAG = FacultyTag(key="tag-key", value="tag-value")
MLFLOW_TAG = RunTag("tag-key", "tag-value")
