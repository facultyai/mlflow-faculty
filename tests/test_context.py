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

from mlflow_faculty.context import FacultyRunContext


USER_ID = uuid4()


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


def test_tags(mocker):
    mocker.patch(
        "os.environ",
        {
            "FACULTY_PROJECT_ID": "project-id",
            "FACULTY_SERVER_ID": "server-id",
            "FACULTY_SERVER_NAME": "server-name",
            "NUM_CPUS": "4",
            "AVAILABLE_MEMORY_MB": "4000",
            "NUM_GPUS": "1",
            "FACULTY_JOB_ID": "job-id",
            "FACULTY_JOB_NAME": "job-name",
            "FACULTY_RUN_ID": "run-id",
            "FACULTY_RUN_NUMBER": "23",
            "FACULTY_SUBRUN_ID": "subrun-id",
            "FACULTY_SUBRUN_NUMBER": "2",
        },
    )

    mock_client = mocker.Mock()
    mock_client.authenticated_user_id.return_value = USER_ID
    mocker.patch("faculty.client", return_value=mock_client)

    expected_tags = {
        "mlflow.faculty.user.userId": str(USER_ID),
        "mlflow.faculty.project.projectId": "project-id",
        "mlflow.faculty.server.serverId": "server-id",
        "mlflow.faculty.server.name": "server-name",
        "mlflow.faculty.server.cpus": "4",
        "mlflow.faculty.server.memoryMb": "4000",
        "mlflow.faculty.server.gpus": "1",
        "mlflow.faculty.job.jobId": "job-id",
        "mlflow.faculty.job.name": "job-name",
        "mlflow.faculty.job.runId": "run-id",
        "mlflow.faculty.job.runNumber": "23",
        "mlflow.faculty.job.subrunId": "subrun-id",
        "mlflow.faculty.job.subrunNumber": "2",
    }

    assert FacultyRunContext().tags() == expected_tags
