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
from faculty.clients.base import HttpError
from faculty.clients.experiment import Experiment
from mlflow.entities import Experiment as MLExperiment, LifecycleStage
from mlflow.exceptions import MlflowException
import pytest
from pytz import UTC

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
        "faculty:{}".format(PROJECT_ID),
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


def test_init_invalid_uri_scheme():
    store_uri = "invalid-scheme:/{}".format(PROJECT_ID)
    expected_error_message = "Not a faculty URI: {}".format(store_uri)
    with pytest.raises(ValueError, match=expected_error_message):
        FacultyRestStore(store_uri)


def test_init_invalid_uri_in_netloc():
    store_uri = "faculty://{}".format(PROJECT_ID)
    expectred_error_message = "Invalid URI {}. Netloc is reserved. Did you mean 'faculty:/{}".format(
        store_uri, PROJECT_ID
    )
    with pytest.raises(ValueError, match=expectred_error_message):
        FacultyRestStore(store_uri)


def test_init_invalid_uri_bad_uuid():
    store_uri = "faculty:/invalid_uuid"
    expectred_error_message = "invalid_uuid in given URI {} is not a valid UUID".format(
        store_uri
    )
    with pytest.raises(ValueError, match=expectred_error_message):
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
    mock_client.create.side_effect = HttpError(
        mocker.Mock(), "Name already used in project."
    )
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(MlflowException, match="Name already used in project."):
        store.create_experiment(NAME, ARTIFACT_LOCATION)


def test_get_experiment(mocker):
    mock_client = mocker.Mock()
    mock_client.get.return_value = FACULTY_EXPERIMENT
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)
    experiment = store.get_experiment(EXPERIMENT_ID)

    assert experiments_equal(experiment, MLFLOW_EXPERIMENT)
    mock_client.get.assert_called_once_with(PROJECT_ID, EXPERIMENT_ID)


def test_get_experiment_client_error(mocker):
    mock_client = mocker.Mock()
    mock_client.get.side_effect = HttpError(
        mocker.Mock(), "Experiment with ID _ not found in project _"
    )
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(
        MlflowException, match="Experiment with ID _ not found in project _"
    ):
        store.get_experiment(EXPERIMENT_ID)


def test_get_experiment_deleted(mocker):
    mock_client = mocker.Mock()
    mock_client.get.return_value.deleted_at = datetime.now(tz=UTC)
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)
    experiment = store.get_experiment(EXPERIMENT_ID)

    assert experiment.lifecycle_stage == LifecycleStage.DELETED


def test_list_experiments(mocker):
    mock_client = mocker.Mock()
    mock_client.list.return_value = [FACULTY_EXPERIMENT]
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)
    experiments = store.list_experiments(PROJECT_ID)

    assert len(experiments) == 1
    assert experiments_equal(experiments[0], MLFLOW_EXPERIMENT)
    mock_client.list.assert_called_once_with(PROJECT_ID)


def test_list_experiments_client_error(mocker):
    mock_client = mocker.Mock()
    mock_client.list.side_effect = HttpError(mocker.Mock(), "Error")
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(MlflowException, match="Error"):
        store.list_experiments(PROJECT_ID)


def test_create_run(mocker):
    mock_client = mocker.Mock()
    mock_client.create_run.return_value = None
    mocker.patch("faculty.client", return_value=mock_client)
    mock_mlflow_run = mocker.Mock()
    mocker.patch(
        "mlflow_faculty.trackingstore.faculty_run_to_mlflow_run",
        return_value=mock_mlflow_run,
    )

    start_time = datetime.now()

    store = FacultyRestStore(STORE_URI)

    run = store.create_run(
        FACULTY_EXPERIMENT.id,
        "mlflow-user-id",
        "run-name",
        "source-type",
        "source-name",
        "entry-point-name",
        start_time.timestamp(),
        "source-version",
        list(),
        "parent-run-id",
    )
    assert run == mock_mlflow_run
    mock_client.create_run.assert_called_once_with(
        PROJECT_ID, FACULTY_EXPERIMENT.id, start_time
    )


def test_create_run_client_error(mocker):
    mock_client = mocker.Mock()
    mock_client.create_run.side_effect = HttpError(mocker.Mock(), "Some error")
    mocker.patch("faculty.client", return_value=mock_client)
    store = FacultyRestStore(STORE_URI)

    with pytest.raises(MlflowException, match="Some error"):
        store.create_run(
            FACULTY_EXPERIMENT.id,
            "mlflow-user-id",
            "run-name",
            "source-type",
            "source-name",
            "entry-point-name",
            datetime.now().timestamp(),
            "source-version",
            list(),
            "parent-run-id",
        )
