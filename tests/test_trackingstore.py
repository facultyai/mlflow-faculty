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
import time
from pytz import UTC

import faculty
from faculty.clients.base import HttpError
from faculty.clients.experiment import (
    ListExperimentRunsResponse,
    Pagination,
    Page,
)
from mlflow.entities import LifecycleStage
from mlflow.exceptions import MlflowException
import pytest

from mlflow_faculty.trackingstore import FacultyRestStore
from mlflow_faculty.py23 import to_timestamp
from tests.fixtures import (
    ARTIFACT_LOCATION,
    EXPERIMENT_ID,
    EXPERIMENT_RUN_UUID,
    EXPERIMENT_RUN_UUID_HEX_STR,
    FACULTY_EXPERIMENT,
    FACULTY_RUN,
    FACULTY_TAG,
    NAME,
    MLFLOW_EXPERIMENT,
    MLFLOW_METRIC,
    MLFLOW_PARAM,
    MLFLOW_TAG,
    PROJECT_ID,
    STORE_URI,
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
    expected_error_message = (
        "Invalid URI {}. Netloc is reserved. "
        "Did you mean 'faculty:{}".format(store_uri, PROJECT_ID)
    )
    with pytest.raises(ValueError, match=expected_error_message):
        FacultyRestStore(store_uri)


def test_init_invalid_uri_bad_uuid():
    store_uri = "faculty:/invalid_uuid"
    expected_error = "invalid_uuid in given URI {} is not a valid UUID".format(
        store_uri
    )
    with pytest.raises(ValueError, match=expected_error):
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
    experiments = store.list_experiments()

    assert len(experiments) == 1
    assert experiments_equal(experiments[0], MLFLOW_EXPERIMENT)
    mock_client.list.assert_called_once_with(PROJECT_ID)


def test_list_experiments_client_error(mocker):
    mock_client = mocker.Mock()
    mock_client.list.side_effect = HttpError(mocker.Mock(), "Error")
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(MlflowException, match="Error"):
        store.list_experiments()


def test_create_run(mocker):
    mock_client = mocker.Mock()
    mock_run = mocker.Mock()
    mock_client.create_run.return_value = mock_run
    mocker.patch("faculty.client", return_value=mock_client)
    mock_mlflow_run = mocker.Mock()
    converter_mock = mocker.patch(
        "mlflow_faculty.trackingstore.faculty_run_to_mlflow_run",
        return_value=mock_mlflow_run,
    )

    # this is how Mlflow creates the start time
    start_time = time.time() * 1000
    expected_start_time = datetime.fromtimestamp(start_time / 1000, tz=UTC)

    store = FacultyRestStore(STORE_URI)

    run = store.create_run(
        FACULTY_EXPERIMENT.id,
        "mlflow-user-id",
        "run-name",
        "source-type",
        "source-name",
        "entry-point-name",
        start_time,
        "source-version",
        [MLFLOW_TAG],
        "parent-run-id",
    )
    assert run == mock_mlflow_run
    mock_client.create_run.assert_called_once_with(
        PROJECT_ID,
        FACULTY_EXPERIMENT.id,
        expected_start_time,
        tags=[FACULTY_TAG],
    )
    converter_mock.assert_called_once_with(mock_run)


def test_create_run_no_tags(mocker):
    mock_client = mocker.Mock()
    mock_run = mocker.Mock()
    mock_client.create_run.return_value = mock_run
    mocker.patch("faculty.client", return_value=mock_client)
    mock_mlflow_run = mocker.Mock()
    converter_mock = mocker.patch(
        "mlflow_faculty.trackingstore.faculty_run_to_mlflow_run",
        return_value=mock_mlflow_run,
    )

    # this is how Mlflow creates the start time
    start_time = time.time() * 1000
    expected_start_time = datetime.fromtimestamp(start_time / 1000, tz=UTC)

    store = FacultyRestStore(STORE_URI)

    run = store.create_run(
        FACULTY_EXPERIMENT.id,
        "mlflow-user-id",
        "run-name",
        "source-type",
        "source-name",
        "entry-point-name",
        start_time,
        "source-version",
        [],
        "parent-run-id",
    )
    assert run == mock_mlflow_run
    mock_client.create_run.assert_called_once_with(
        PROJECT_ID, FACULTY_EXPERIMENT.id, expected_start_time, tags=[]
    )
    converter_mock.assert_called_once_with(mock_run)


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
            to_timestamp(datetime.now(tz=UTC)),
            "source-version",
            list(),
            "parent-run-id",
        )


def test_get_run(mocker):
    mock_client = mocker.Mock()
    mock_client.get_run.return_value = FACULTY_RUN
    mocker.patch("faculty.client", return_value=mock_client)
    mock_mlflow_run = mocker.Mock()
    converter_mock = mocker.patch(
        "mlflow_faculty.trackingstore.faculty_run_to_mlflow_run",
        return_value=mock_mlflow_run,
    )

    store = FacultyRestStore(STORE_URI)
    run = store.get_run(EXPERIMENT_RUN_UUID_HEX_STR)

    assert run == mock_mlflow_run

    mock_client.get_run.assert_called_once_with(
        PROJECT_ID, EXPERIMENT_RUN_UUID
    )
    converter_mock.assert_called_once_with(FACULTY_RUN)


def test_get_run_client_error(mocker):
    mock_client = mocker.Mock()
    mock_client.get_run.side_effect = HttpError(
        mocker.Mock(), "Experiment run with ID _ not found in project _"
    )
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(
        MlflowException,
        match="Experiment run with ID _ not found in project _",
    ):
        store.get_run(EXPERIMENT_RUN_UUID_HEX_STR)


def test_search_runs(mocker):
    mock_faculty_runs = [mocker.Mock(), mocker.Mock(), mocker.Mock()]
    list_page_1 = ListExperimentRunsResponse(
        runs=[mock_faculty_runs[0], mock_faculty_runs[1]],
        pagination=Pagination(
            start=0, size=2, previous=None, next=Page(start=2, limit=1)
        ),
    )
    list_page_2 = ListExperimentRunsResponse(
        runs=[mock_faculty_runs[2]],
        pagination=Pagination(
            start=0, size=2, previous=Page(start=0, limit=2), next=None
        ),
    )

    mock_client = mocker.Mock()
    mock_client.list_runs.side_effect = [list_page_1, list_page_2]
    mocker.patch("faculty.client", return_value=mock_client)

    mock_mlflow_runs = [mocker.Mock(), mocker.Mock(), mocker.Mock()]
    converter_mock = mocker.patch(
        "mlflow_faculty.trackingstore.faculty_run_to_mlflow_run",
        side_effect=mock_mlflow_runs,
    )

    store = FacultyRestStore(STORE_URI)
    runs = store.search_runs(
        experiment_ids=None, search_expressions=None, run_view_type=None
    )

    assert runs == mock_mlflow_runs
    mock_client.list_runs.assert_has_calls(
        [
            mocker.call(PROJECT_ID, experiment_ids=None),
            mocker.call(PROJECT_ID, experiment_ids=None),
        ]
    )
    converter_mock.assert_has_calls(
        [
            mocker.call(mock_faculty_runs[0]),
            mocker.call(mock_faculty_runs[1]),
            mocker.call(mock_faculty_runs[2]),
        ]
    )


def test_search_runs_empty_page(mocker):
    list_page = ListExperimentRunsResponse(
        runs=[],
        pagination=Pagination(start=0, size=0, previous=None, next=None),
    )

    mock_client = mocker.Mock()
    mock_client.list_runs.side_effect = [list_page]
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)
    runs = store.search_runs(
        experiment_ids=None, search_expressions=None, run_view_type=None
    )

    assert runs == []
    mock_client.list_runs.assert_called_once_with(
        PROJECT_ID, experiment_ids=None
    )


def test_search_runs_next_page_but_no_runs(mocker):
    list_page = ListExperimentRunsResponse(
        runs=[],
        pagination=Pagination(
            start=0, size=0, previous=None, next=Page(start=999, limit=999)
        ),
    )

    mock_client = mocker.Mock()
    mock_client.list_runs.side_effect = [list_page]
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)
    runs = store.search_runs(
        experiment_ids=None, search_expressions=None, run_view_type=None
    )

    assert runs == []
    mock_client.list_runs.assert_called_once_with(
        PROJECT_ID, experiment_ids=None
    )


def test_search_runs_filter_by_experiment(mocker):
    list_page = ListExperimentRunsResponse(
        runs=[],
        pagination=Pagination(start=0, size=0, previous=None, next=None),
    )

    mock_client = mocker.Mock()
    mock_client.list_runs.side_effect = [list_page]
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)
    runs = store.search_runs(
        experiment_ids=[123, 456], search_expressions=None, run_view_type=None
    )

    assert runs == []
    mock_client.list_runs.assert_called_once_with(
        PROJECT_ID, experiment_ids=[123, 456]
    )


def test_search_runs_client_error(mocker):
    mock_client = mocker.Mock()
    mock_client.list_runs.side_effect = HttpError(
        mocker.Mock(), "Dummy client error."
    )
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(MlflowException, match="Dummy client error."):
        store.search_runs(
            [FACULTY_EXPERIMENT.id],
            search_expressions=None,
            run_view_type=None,
        )


def test_log_batch(mocker):
    mock_client = mocker.Mock()
    mocker.patch("faculty.client", return_value=mock_client)
    mock_mlflow_metric = mocker.Mock()
    metric_converter_mock = mocker.patch(
        "mlflow_faculty.trackingstore.mlflow_metrics_to_faculty_metrics",
        return_value=mock_mlflow_metric,
    )
    mock_mlflow_param = mocker.Mock()
    param_converter_mock = mocker.patch(
        "mlflow_faculty.trackingstore.mlflow_params_to_faculty_params",
        return_value=mock_mlflow_param,
    )
    mock_mlflow_tag = mocker.Mock()
    tag_converter_mock = mocker.patch(
        "mlflow_faculty.trackingstore.mlflow_tags_to_faculty_tags",
        return_value=mock_mlflow_tag,
    )

    store = FacultyRestStore(STORE_URI)
    store.log_batch(
        EXPERIMENT_RUN_UUID_HEX_STR,
        metrics=[MLFLOW_METRIC],
        params=[MLFLOW_PARAM],
        tags=[MLFLOW_TAG],
    )

    mock_client.log_run_data.assert_called_once_with(
        PROJECT_ID,
        EXPERIMENT_RUN_UUID,
        metrics=[metric_converter_mock.return_value],
        params=[param_converter_mock.return_value],
        tags=[tag_converter_mock.return_value],
    )
    metric_converter_mock.assert_called_once_with(MLFLOW_METRIC)
    param_converter_mock.assert_called_once_with(MLFLOW_PARAM)
    tag_converter_mock.assert_called_once_with(MLFLOW_TAG)


def test_log_batch_empty(mocker):
    mock_client = mocker.Mock()
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)
    store.log_batch(EXPERIMENT_RUN_UUID_HEX_STR)

    mock_client.log_run_data.assert_called_once_with(
        PROJECT_ID, EXPERIMENT_RUN_UUID, metrics=[], params=[], tags=[]
    )


def test_log_batch_param_conflict(mocker):
    mock_client = mocker.Mock()
    exception = faculty.clients.experiment.ParamConflict(
        message="message", conflicting_params=["param-key"]
    )

    mock_client.log_run_data.side_effect = exception
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(MlflowException, match="param-key"):
        with pytest.raises(
            faculty.clients.experiment.ParamConflict, match="message"
        ):
            store.log_batch(EXPERIMENT_RUN_UUID_HEX_STR)
    mock_client.log_run_data.assert_called_once_with(
        PROJECT_ID, EXPERIMENT_RUN_UUID, metrics=[], params=[], tags=[]
    )


def test_log_batch_error(mocker):
    mock_client = mocker.Mock()
    exception = HttpError(
        mocker.Mock(), error="error_message", error_code="some_error_code"
    )

    mock_client.log_run_data.side_effect = exception
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(MlflowException, match="error_message"):
        with pytest.raises(HttpError, match="some_error_code"):
            store.log_batch(EXPERIMENT_RUN_UUID_HEX_STR)
    mock_client.log_run_data.assert_called_once_with(
        PROJECT_ID, EXPERIMENT_RUN_UUID, metrics=[], params=[], tags=[]
    )
