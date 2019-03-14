from copy import copy
from datetime import datetime
from uuid import uuid4

import pytest
from pytz import UTC

from faculty.clients.experiment import (
    ExperimentRunStatus as FacultyExperimentRunStatus,
    ExperimentRun as FacultyExperimentRun,
)

from mlflow.entities import LifecycleStage, Run, RunData, RunInfo, RunStatus
from mlflow_faculty.mlflow_converters import (
    faculty_run_to_mlflow_run,
    mlflow_run_metric_to_faculty_run_metric,
    mlflow_run_param_to_faculty_run_param,
    mlflow_run_tag_to_faculty_run_tag,
    mlflow_timestamp_to_datetime_milliseconds,
    mlflow_timestamp_to_datetime_seconds,
)
from mlflow_faculty.py23 import to_timestamp

from tests.fixtures import (
    FACULTY_METRIC,
    FACULTY_PARAM,
    FACULTY_TAG,
    MLFLOW_METRIC,
    MLFLOW_PARAM,
    MLFLOW_TAG,
)

EXPERIMENT_RUN_UUID = uuid4()
EXPERIMENT_RUN_UUID_HEX_STR = EXPERIMENT_RUN_UUID.hex
RUN_STARTED_AT = datetime(2018, 3, 10, 11, 39, 12, 110000, tzinfo=UTC)
RUN_STARTED_AT_INT = to_timestamp(RUN_STARTED_AT) * 1000

FACULTY_RUN = FacultyExperimentRun(
    id=EXPERIMENT_RUN_UUID,
    experiment_id=661,
    artifact_location="faculty:",
    status=FacultyExperimentRunStatus.RUNNING,
    started_at=RUN_STARTED_AT,
    ended_at=None,
    deleted_at=None,
)


EXPECTED_RUN_INFO = RunInfo(
    EXPERIMENT_RUN_UUID_HEX_STR,
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
    ],
)
def test_mlflow_timestamp_to_datetime_milliseconds(
    timestamp, expected_datetime
):
    assert (
        mlflow_timestamp_to_datetime_milliseconds(timestamp)
        == expected_datetime
    )


@pytest.mark.parametrize(
    "timestamp, expected_datetime",
    [
        (0, datetime(1970, 1, 1, tzinfo=UTC)),
        (1552484641, datetime(2019, 3, 13, 13, 44, 1, tzinfo=UTC)),
    ],
)
def test_mlflow_timestamp_to_datetime_seconds(timestamp, expected_datetime):
    assert mlflow_timestamp_to_datetime_seconds(timestamp) == expected_datetime


def test_mlflow_run_metric_to_faculty_run_metric():
    assert (
        mlflow_run_metric_to_faculty_run_metric(MLFLOW_METRIC)
        == FACULTY_METRIC
    )


def test_mlflow_run_param_to_faculty_run_param():
    assert mlflow_run_param_to_faculty_run_param(MLFLOW_PARAM) == FACULTY_PARAM


def test_mlflow_run_tag_to_faculty_run_tag():
    assert mlflow_run_tag_to_faculty_run_tag(MLFLOW_TAG) == FACULTY_TAG
