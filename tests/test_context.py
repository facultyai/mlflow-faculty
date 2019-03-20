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

from uuid import uuid4

import pytest
import mock

from mlflow_faculty.context import FacultyRunContext


def merge_dicts(*dicts):
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


MOCK_ACCOUNT = mock.Mock(user_id=uuid4(), username="joe_bloggs")

STANDARD_ENVIRONMENT = {
    "FACULTY_PROJECT_ID": "project-id",
    "FACULTY_SERVER_ID": "server-id",
    "FACULTY_SERVER_NAME": "server-name",
    "FACULTY_SERVER_TYPE": "jupyter",
    "NUM_CPUS": "4",
    "AVAILABLE_MEMORY_MB": "4000",
    "NUM_GPUS": "1",
}
JOB_ENVIRONMENT = {
    "FACULTY_SERVER_TYPE": "python-job",
    "FACULTY_JOB_ID": "job-id",
    "FACULTY_JOB_NAME": "job-name",
    "FACULTY_RUN_ID": "run-id",
    "FACULTY_RUN_NUMBER": "23",
    "FACULTY_SUBRUN_ID": "subrun-id",
    "FACULTY_SUBRUN_NUMBER": "2",
}

ALL_ENVIRONMENT_EMPTY_STRING = {
    key: "" for key in merge_dicts(STANDARD_ENVIRONMENT, JOB_ENVIRONMENT)
}

DEFAULT_TAGS = {"mlflow.faculty.createdBy": "user"}
STANDARD_TAGS = {
    "mlflow.faculty.createdBy": "user",
    "mlflow.faculty.project.projectId": "project-id",
    "mlflow.faculty.server.serverId": "server-id",
    "mlflow.faculty.server.name": "server-name",
    "mlflow.faculty.server.cpus": "4",
    "mlflow.faculty.server.memoryMb": "4000",
    "mlflow.faculty.server.gpus": "1",
}
USER_TAGS = {
    "mlflow.faculty.user.userId": str(MOCK_ACCOUNT.user_id),
    "mlflow.faculty.user.username": MOCK_ACCOUNT.username,
}
JOB_TAGS = {
    "mlflow.faculty.createdBy": "job",
    "mlflow.faculty.job.jobId": "job-id",
    "mlflow.faculty.job.name": "job-name",
    "mlflow.faculty.job.runId": "run-id",
    "mlflow.faculty.job.runNumber": "23",
    "mlflow.faculty.job.subrunId": "subrun-id",
    "mlflow.faculty.job.subrunNumber": "2",
}


@pytest.mark.parametrize(
    "env, in_context",
    [
        ({"FACULTY_PROJECT_ID": "project-id"}, True),
        ({"FACULTY_PROJECT_ID": ""}, False),
        ({}, False),
    ],
)
def test_in_context(mocker, env, in_context):
    mocker.patch("os.environ", env)
    assert FacultyRunContext().in_context() is in_context


@pytest.mark.parametrize(
    "environment, expected_tags",
    [
        (STANDARD_ENVIRONMENT, merge_dicts(STANDARD_TAGS, USER_TAGS)),
        (
            merge_dicts(STANDARD_ENVIRONMENT, JOB_ENVIRONMENT),
            merge_dicts(STANDARD_TAGS, USER_TAGS, JOB_TAGS),
        ),
        ({}, merge_dicts(DEFAULT_TAGS, USER_TAGS)),
        (ALL_ENVIRONMENT_EMPTY_STRING, merge_dicts(DEFAULT_TAGS, USER_TAGS)),
    ],
    ids=["standard", "job", "no-env", "env-empty-string"],
)
def test_tags(mocker, environment, expected_tags):
    mocker.patch("os.environ", environment)

    mock_client = mocker.Mock()
    mock_client.authenticated_account.return_value = MOCK_ACCOUNT
    mocker.patch("faculty.client", return_value=mock_client)

    assert FacultyRunContext().tags() == expected_tags


@pytest.mark.parametrize(
    "environment, expected_tags",
    [
        (STANDARD_ENVIRONMENT, STANDARD_TAGS),
        (
            merge_dicts(STANDARD_ENVIRONMENT, JOB_ENVIRONMENT),
            merge_dicts(STANDARD_TAGS, JOB_TAGS),
        ),
        ({}, DEFAULT_TAGS),
        (ALL_ENVIRONMENT_EMPTY_STRING, DEFAULT_TAGS),
    ],
    ids=["standard", "job", "no-env", "env-empty-string"],
)
def test_tags_error_getting_account(mocker, environment, expected_tags):
    mocker.patch("os.environ", environment)

    mock_client = mocker.Mock()
    mock_client.authenticated_account.side_effect = Exception()
    mocker.patch("faculty.client", return_value=mock_client)

    assert FacultyRunContext().tags() == expected_tags
