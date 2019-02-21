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

import pytest
from pytz import UTC
import faculty
from faculty.clients.base import HttpError
from faculty.clients.experiment import Experiment
from mlflow.entities import Experiment as MLExperiment, LifecycleStage
from mlflow_faculty.trackingstore import FacultyRestStore

PROJECT_ID = uuid4()
STORE_URI = "faculty:{}".format(PROJECT_ID)

EXPERIMENT_ID = 12
CREATED_AT = datetime.now(tz=UTC)
LAST_UPDATED_AT = CREATED_AT

NAME = "experiment name"
ARTIFACT_LOCATION = "scheme://artifact-location"

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


def experiments_equal(one, two):
    return (
        one.experiment_id == two.experiment_id
        and one.name == two.name
        and one.artifact_location == two.artifact_location
        and one.lifecycle_stage == two.lifecycle_stage
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
    experiment_id = store.create_experiment(NAME, ARTIFACT_LOCATION)

    assert experiment_id == FACULTY_EXPERIMENT.id
    mock_client.create.assert_called_once_with(
        PROJECT_ID, NAME, artifact_location=ARTIFACT_LOCATION
    )


def test_create_experiment_client_error(mocker):
    mock_client = mocker.Mock()
    mock_client.create.side_effect = HttpError(mocker.Mock(), mocker.Mock())
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    returned_experiment_id = store.create_experiment(NAME, ARTIFACT_LOCATION)
    assert returned_experiment_id is None


def test_get_experiment(mocker):
    mock_client = mocker.Mock()
    mock_client.get.return_value = FACULTY_EXPERIMENT
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)
    experiment = store.get_experiment(EXPERIMENT_ID)

    assert experiments_equal(experiment, MLFLOW_EXPERIMENT)
    mock_client.get.assert_called_once_with(PROJECT_ID, EXPERIMENT_ID)


def test_get_experiment_deleted(mocker):
    mock_client = mocker.Mock()
    mock_client.get.return_value.deleted_at = datetime.now(tz=UTC)
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)
    experiment = store.get_experiment(EXPERIMENT_ID)

    assert experiment.lifecycle_stage == LifecycleStage.DELETED
