from datetime import datetime
from uuid import uuid4

from pytz import UTC

from mlflow_faculty.mlflow_converters import faculty_run_to_mlflow_run
from mlflow.entities import RunInfo, RunStatus, LifecycleStage

from faculty.clients.experiment import ExperimentRun as FacultyExperimentRun, ExperimentRunStatus as FacultyExperimentRunStatus

EXPERIMENT_RUN_ID = uuid4()
RUN_STARTED_AT = datetime(2018, 3, 10, 11, 39, 12, 110000, tzinfo=UTC)

FACULTY_RUN = FacultyExperimentRun(
    id=EXPERIMENT_RUN_ID,
    experiment_id=661,
    artifact_location="faculty:",
    status=FacultyExperimentRunStatus.RUNNING,
    started_at=RUN_STARTED_AT,
    ended_at=None,
    deleted_at=None
)


EXPECTED_RUN_INFO = RunInfo(
    EXPERIMENT_RUN_ID,
    661,
    "",  # name
    "",  # source_type
    "",  # source_name
    "",  # entry_point_name
    "",  # user_id
    RunStatus.RUNNING,
    RUN_STARTED_AT,
    None,
    "",  # source_version
    LifecycleStage.ACTIVE
)


def test_convert_run():
    assert faculty_run_to_mlflow_run(FACULTY_RUN) == EXPECTED_RUN_INFO
