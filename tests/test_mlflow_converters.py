from copy import copy
from datetime import datetime
from uuid import uuid4

from pytz import UTC

import pytest

from mlflow.entities import Run, RunData, RunInfo, RunStatus, LifecycleStage

from faculty.clients.experiment import (
    ExperimentRun as FacultyExperimentRun,
    ExperimentRunStatus as FacultyExperimentRunStatus,
)

from mlflow_faculty.mlflow_converters import (
    faculty_run_to_mlflow_run,
    mlflow_timestamp_to_datetime
)
from mlflow_faculty.py23 import to_timestamp

EXPERIMENT_RUN_ID = uuid4()
RUN_STARTED_AT = datetime(2018, 3, 10, 11, 39, 12, 110000, tzinfo=UTC)
RUN_STARTED_AT_INT = to_timestamp(RUN_STARTED_AT) * 1000

FACULTY_RUN = FacultyExperimentRun(
    id=EXPERIMENT_RUN_ID,
    experiment_id=661,
    artifact_location="faculty:",
    status=FacultyExperimentRunStatus.RUNNING,
    started_at=RUN_STARTED_AT,
    ended_at=None,
    deleted_at=None,
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
    RUN_STARTED_AT_INT,
    None,
    "",  # source_version
    LifecycleStage.ACTIVE,
)
EXPECTED_RUN = Run(EXPECTED_RUN_INFO, RunData())


def check_run_data_equals(first, other):
    return (
        first.metrics == other.metrics
        and first.params == other.params
        and first.tags == other.tags
    )


def check_run_equals(first, other):
    # return first.info == other.info
    return (
        check_run_data_equals(first.data, other.data)
        and first.info == other.info
    )


def test_convert_run():
    assert check_run_equals(
        faculty_run_to_mlflow_run(FACULTY_RUN), EXPECTED_RUN
    )


@pytest.mark.parametrize(
    "faculty_run_status, run_status",
    [
        (FacultyExperimentRunStatus.RUNNING, RunStatus.RUNNING),
        (FacultyExperimentRunStatus.FINISHED, RunStatus.FINISHED),
        (FacultyExperimentRunStatus.FAILED, RunStatus.FAILED),
        (FacultyExperimentRunStatus.SCHEDULED, RunStatus.SCHEDULED),
    ],
)
def test_convert_run_status(faculty_run_status, run_status):
    faculty_run = FACULTY_RUN._replace(status=faculty_run_status)
    expected_run_info = copy(EXPECTED_RUN_INFO)
    expected_run_info._status = run_status
    expected_run = Run(expected_run_info, RunData())
    assert check_run_equals(
        faculty_run_to_mlflow_run(faculty_run), expected_run
    )


def test_deleted_runs():
    faculty_run = FACULTY_RUN._replace(deleted_at=datetime.now())
    expected_run_info = copy(EXPECTED_RUN_INFO)
    expected_run_info._lifecycle_stage = LifecycleStage.DELETED
    expected_run = Run(expected_run_info, RunData())
    assert check_run_equals(
        faculty_run_to_mlflow_run(faculty_run), expected_run
    )


def test_run_end_time():
    ended_at = datetime.now(tz=UTC)
    faculty_run = FACULTY_RUN._replace(ended_at=ended_at)
    expected_run_info = copy(EXPECTED_RUN_INFO)
    expected_run_info._end_time = to_timestamp(ended_at) * 1000
    expected_run = Run(expected_run_info, RunData())
    assert check_run_equals(
        faculty_run_to_mlflow_run(faculty_run), expected_run
    )


@pytest.mark.parametrize(
    "timestamp, expected_datetime",
    [
        (0, datetime(1970, 1, 1, tzinfo=UTC)),
        (1551884271987, datetime(2019, 3, 6, 14, 57, 51, 987000, tzinfo=UTC)),
    ]
)
def test_mlflow_timestamp_to_datetime(timestamp, expected_datetime):
    assert mlflow_timestamp_to_datetime(timestamp) == expected_datetime
