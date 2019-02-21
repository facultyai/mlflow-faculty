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

import faculty
from faculty.clients.experiment import Experiment
from faculty.clients.base import HttpError
from mlflow.entities import Experiment as MLExperiment
from mlflow_faculty.trackingstore import FacultyRestStore
from pytz import UTC
import pytest

PROJECT_ID = uuid4()
STORE_URI = "faculty:{}".format(PROJECT_ID)
EXPERIMENT_ID = 12345
CREATED_AT = datetime(2018, 3, 10, 11, 32, 6, 247000, tzinfo=UTC)
LAST_UPDATED_AT = datetime(2018, 3, 10, 11, 32, 30, 172000, tzinfo=UTC)
DELETED_AT = datetime(2018, 3, 10, 11, 37, 42, 482000, tzinfo=UTC)

MLFLOW_EXPERIMENT = MLExperiment(
    EXPERIMENT_ID, "test-name", "file://test", "active"
)

FACULTY_EXPERIMENT = Experiment(
    id=12,
    name="experiment name",
    description="experiment description",
    artifact_location="https://example.com",
    created_at=CREATED_AT,
    last_updated_at=LAST_UPDATED_AT,
    deleted_at=DELETED_AT,
)


@pytest.mark.parametrize(
    "store_uri",
    [
        STORE_URI,
        "faculty:/{}".format(PROJECT_ID),
        "faculty:///{}".format(PROJECT_ID),
        "faculty:///{}/".format(PROJECT_ID),
    ],
)
def test_init(mocker, store_uri):
    mocker.patch("faculty.client")

    store = FacultyRestStore(store_uri)

    assert store._project_id == PROJECT_ID
    faculty.client.assert_called_once_with("experiment")


@pytest.mark.parametrize(
    "store_uri",
    [
        "invalid-scheme://{}".format(PROJECT_ID),
        "faculty://{}".format(PROJECT_ID),
        "faculty:invalid-uuid",
    ],
)
def test_init_invalid_uri(store_uri):
    with pytest.raises(ValueError):
        FacultyRestStore(store_uri)


def test_create_experiment(mocker):
    mock_client = mocker.Mock()
    mock_client.create.return_value = FACULTY_EXPERIMENT
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)
    returned_experiment_id = store.create_experiment(
        MLFLOW_EXPERIMENT.name, MLFLOW_EXPERIMENT.artifact_location
    )

    assert returned_experiment_id == FACULTY_EXPERIMENT.id
    mock_client.create.assert_called_once_with(
        PROJECT_ID,
        MLFLOW_EXPERIMENT.name,
        artifact_location=MLFLOW_EXPERIMENT.artifact_location,
    )


def test_create_experiment_client_error(mocker):
    mock_client = mocker.Mock()
    mock_client.create.side_effect = HttpError(mocker.Mock(), mocker.Mock())
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    returned_experiment_id = store.create_experiment(
        MLFLOW_EXPERIMENT.name, MLFLOW_EXPERIMENT.artifact_location
    )
    assert returned_experiment_id is None
