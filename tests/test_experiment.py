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
# limitations under the License

from datetime import datetime
from uuid import uuid4

from mlflow.entities import Experiment
from mlflow_faculty.trackingstore import FacultyRestStore
from pytz import UTC

PROJECT_ID = uuid4()
EXPERIMENT_ID = 12345
CREATED_AT = datetime(2018, 3, 10, 11, 32, 6, 247000, tzinfo=UTC)
CREATED_AT_STRING = "2018-03-10T11:32:06.247Z"
LAST_UPDATED_AT = datetime(2018, 3, 10, 11, 32, 30, 172000, tzinfo=UTC)
LAST_UPDATED_AT_STRING = "2018-03-10T11:32:30.172Z"
DELETED_AT = datetime(2018, 3, 10, 11, 37, 42, 482000, tzinfo=UTC)
DELETED_AT_STRING = "2018-03-10T11:37:42.482Z"

MLFLOW_EXPERIMENT = Experiment(
    12345,
    "test-name",
    "file://test",
    "active"
)

FACULTY_EXPERIMENT_BODY = {
    "experimentId": MLFLOW_EXPERIMENT.experiment_id,
    "name": MLFLOW_EXPERIMENT.name,
    "description": "",
    "artifactLocation": MLFLOW_EXPERIMENT.artifact_location,
    "createdAt": CREATED_AT_STRING,
    "lastUpdatedAt": LAST_UPDATED_AT_STRING,
    "deletedAt": DELETED_AT_STRING,
}

# def test_client(mocker):
#     get_session_mock = mocker.patch("faculty.session.get_session")
#     for_resource_mock = mocker.patch("faculty.clients.for_resource")

#     options = {"foo": "bar"}
#     faculty.client("experiment", **options)

#     get_session_mock.assert_called_once_with(**options)
#     for_resource_mock.assert_called_once_with("experiment")

#     returned_session = get_session_mock.return_value
#     returned_class = for_resource_mock.return_value
#     returned_class.assert_called_once_with(returned_session)


def test_experiment_client_get(mocker):

    # mocker.patch.object(, "_post", return_value=EXPERIMENT)

    faculty_client_mock = mocker.patch("faculty.client")
    faculty_client_mock.create.return_value = FACULTY_EXPERIMENT_BODY

    env_mock = mocker.patch(
        "os.getenv",
        return_value=PROJECT_ID
    )

    client = FacultyRestStore(mocker.Mock())
    returned_experiment = client.create_experiment(
        MLFLOW_EXPERIMENT.name,
        MLFLOW_EXPERIMENT.artifact_location
    )

    assert returned_experiment == MLFLOW_EXPERIMENT

    env_mock.assert_called_once_with("FACULTY_PROJECT_ID")
    faculty_client_mock.assert_called_once_with(
        PROJECT_ID,
        MLFLOW_EXPERIMENT.name,
        "",
        MLFLOW_EXPERIMENT.artifact_location
    )
