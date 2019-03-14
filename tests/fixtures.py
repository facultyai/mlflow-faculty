from datetime import datetime
from pytz import UTC
from uuid import uuid4

from mlflow.entities import (
    Experiment as MLExperiment,
    LifecycleStage,
    Metric,
    Param,
    Run,
    RunTag,
    RunData,
    RunInfo,
    RunStatus,
)
from faculty.clients.experiment import (
    Experiment,
    ExperimentRun,
    ExperimentRunStatus,
    Metric as FacultyMetric,
    Param as FacultyParam,
    Tag as FacultyTag,
)
from mlflow_faculty.py23 import to_timestamp


PROJECT_ID = uuid4()
STORE_URI = "faculty:{}".format(PROJECT_ID)

EXPERIMENT_ID = 12
CREATED_AT = datetime.now(tz=UTC)
LAST_UPDATED_AT = CREATED_AT

NAME = "experiment name"
ARTIFACT_LOCATION = "scheme://artifact-location"

METRIC_TIMESTAMP = datetime(2019, 3, 13, 17, 0, 15, tzinfo=UTC)
FACULTY_METRIC = FacultyMetric(
    key="metric-key", value="metric-value", timestamp=METRIC_TIMESTAMP
)
MLFLOW_METRIC = Metric(
    "metric-key", "metric-value", int(METRIC_TIMESTAMP.strftime("%s"))
)

FACULTY_PARAM = FacultyParam(key="param-key", value="param-value")
MLFLOW_PARAM = Param("param-key", "param-value")

FACULTY_TAG = FacultyTag(key="tag-key", value="tag-value")
MLFLOW_TAG = RunTag("tag-key", "tag-value")

MLFLOW_EXPERIMENT = MLExperiment(
    EXPERIMENT_ID, NAME, ARTIFACT_LOCATION, LifecycleStage.ACTIVE
)

FACULTY_EXPERIMENT = Experiment(
    id=EXPERIMENT_ID,
    name=NAME,
    description="not used",
    artifact_location=ARTIFACT_LOCATION,
    created_at=datetime.now(tz=UTC),
    last_updated_at=datetime.now(tz=UTC),
    deleted_at=None,
)

EXPERIMENT_RUN_UUID = uuid4()
EXPERIMENT_RUN_UUID_HEX_STR = EXPERIMENT_RUN_UUID.hex

RUN_STARTED_AT = datetime(2018, 3, 10, 11, 39, 12, 110000, tzinfo=UTC)
RUN_STARTED_AT_INT = to_timestamp(RUN_STARTED_AT) * 1000

FACULTY_RUN = ExperimentRun(
    id=EXPERIMENT_RUN_UUID,
    experiment_id=FACULTY_EXPERIMENT.id,
    artifact_location="faculty:",
    status=ExperimentRunStatus.RUNNING,
    started_at=RUN_STARTED_AT,
    ended_at=None,
    deleted_at=None,
    tags=[FACULTY_TAG],
    params=[],
    metrics=[],
)


MLFLOW_RUN_DATA = RunData(tags=[MLFLOW_TAG])

MLFLOW_RUN_INFO = RunInfo(
    EXPERIMENT_RUN_UUID_HEX_STR,
    EXPERIMENT_ID,
    "",  # name
    "",  # source_type
    "",  # source_name
    "",  # entry_point_name
    "",  # user_id
    RunStatus.RUNNING,
    RUN_STARTED_AT_INT,
    None,
    "",  # source_version
    LifecycleStage.ACTIVE,
)
MLFLOW_RUN = Run(MLFLOW_RUN_INFO, MLFLOW_RUN_DATA)
