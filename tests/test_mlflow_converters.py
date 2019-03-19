# Copyright 2019 Faculty Science Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import copy
from datetime import datetime

import pytest
from pytz import UTC
from requests import Response

from faculty.clients.experiment import (
    ExperimentRunStatus as FacultyExperimentRunStatus,
)
from faculty.clients.base import HTTPError
from mlflow.exceptions import MlflowException
from mlflow.entities import LifecycleStage, Run, RunStatus

from mlflow_faculty.mlflow_converters import (
    faculty_run_to_mlflow_run,
    faculty_http_error_to_mlflow_exception,
    mlflow_metric_to_faculty_metric,
    mlflow_param_to_faculty_param,
    mlflow_tag_to_faculty_tag,
    mlflow_timestamp_to_datetime_milliseconds,
    mlflow_timestamp_to_datetime_seconds,
    faculty_tag_to_mlflow_tag,
)
from mlflow_faculty.py23 import to_timestamp
from tests.fixtures import (
    FACULTY_METRIC,
    FACULTY_PARAM,
    FACULTY_RUN,
    FACULTY_TAG,
    MLFLOW_METRIC,
    MLFLOW_PARAM,
    MLFLOW_RUN,
    MLFLOW_RUN_DATA,
    MLFLOW_RUN_INFO,
    MLFLOW_TAG,
)


def check_run_data_equals(first, other):
    return (
        first.metrics == other.metrics
        and first.params == other.params
        and first.tags == other.tags
    )


def check_run_equals(first, other):
    return (
        check_run_data_equals(first.data, other.data)
        and first.info == other.info
    )


def test_convert_run():
    assert check_run_equals(faculty_run_to_mlflow_run(FACULTY_RUN), MLFLOW_RUN)


def test_faculty_http_error_to_mlflow_exception():
    dummy_response = Response()
    dummy_response.status_code = 418
    faculty_http_error = HTTPError(dummy_response, "error", "error_code")

    assert isinstance(
        faculty_http_error_to_mlflow_exception(faculty_http_error),
        MlflowException,
    )


def test_faculty_tags_to_mlflow_tags():
    assert faculty_tag_to_mlflow_tag(FACULTY_TAG) == MLFLOW_TAG


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
    expected_run_info = copy(MLFLOW_RUN_INFO)
    expected_run_info._status = run_status
    expected_run = Run(expected_run_info, MLFLOW_RUN_DATA)
    assert check_run_equals(
        faculty_run_to_mlflow_run(faculty_run), expected_run
    )


def test_deleted_runs():
    faculty_run = FACULTY_RUN._replace(deleted_at=datetime.now())
    expected_run_info = copy(MLFLOW_RUN_INFO)
    expected_run_info._lifecycle_stage = LifecycleStage.DELETED
    expected_run = Run(expected_run_info, MLFLOW_RUN_DATA)
    assert check_run_equals(
        faculty_run_to_mlflow_run(faculty_run), expected_run
    )


def test_run_end_time():
    ended_at = datetime.now(tz=UTC)
    faculty_run = FACULTY_RUN._replace(ended_at=ended_at)
    expected_run_info = copy(MLFLOW_RUN_INFO)
    expected_run_info._end_time = to_timestamp(ended_at) * 1000
    expected_run = Run(expected_run_info, MLFLOW_RUN_DATA)
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


def test_mlflow_metric_to_faculty_metric():
    assert mlflow_metric_to_faculty_metric(MLFLOW_METRIC) == FACULTY_METRIC


def test_mlflow_param_to_faculty_param():
    assert mlflow_param_to_faculty_param(MLFLOW_PARAM) == FACULTY_PARAM


def test_mlflow_tag_to_faculty_tag():
    assert mlflow_tag_to_faculty_tag(MLFLOW_TAG) == FACULTY_TAG
