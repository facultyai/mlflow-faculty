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

import faculty
from faculty.clients.base import HttpError
from faculty.clients.experiment import (
    ExperimentNameConflict,
    ListExperimentRunsResponse,
    Pagination,
    Page,
)
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
import pytest

from mlflow_faculty.trackingstore import FacultyRestStore
from tests.fixtures import (
    ARTIFACT_LOCATION,
    EXPERIMENT_ID,
    RUN_UUID,
    RUN_UUID_HEX_STR,
    FACULTY_EXPERIMENT,
    FACULTY_RUN,
    NAME,
    MLFLOW_METRIC,
    MLFLOW_PARAM,
    MLFLOW_TAG,
    PROJECT_ID,
)


STORE_URI = "faculty:{}".format(PROJECT_ID)


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


def test_create_experiment_name_conflict(mocker):
    exception = ExperimentNameConflict("bad name")
    mock_client = mocker.Mock()
    mock_client.create.side_effect = exception
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(MlflowException, match=str(exception)):
        store.create_experiment(NAME, ARTIFACT_LOCATION)


def test_create_experiment_client_error(mocker):
    mock_client = mocker.Mock()
    mock_client.create.side_effect = HttpError(mocker.Mock(), "Error")
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(MlflowException, match="Error"):
        store.create_experiment(NAME, ARTIFACT_LOCATION)


def test_get_experiment(mocker):
    faculty_experiment = mocker.Mock()
    mock_client = mocker.Mock()
    mock_client.get.return_value = faculty_experiment
    mocker.patch("faculty.client", return_value=mock_client)

    mlflow_experiment = mocker.Mock()
    converter = mocker.patch(
        "mlflow_faculty.trackingstore.faculty_experiment_to_mlflow_experiment",
        return_value=mlflow_experiment,
    )

    store = FacultyRestStore(STORE_URI)
    experiment = store.get_experiment(EXPERIMENT_ID)

    mock_client.get.assert_called_once_with(PROJECT_ID, EXPERIMENT_ID)
    converter.assert_called_once_with(faculty_experiment)
    assert experiment == mlflow_experiment


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


def test_list_experiments(mocker):
    faculty_experiment = mocker.Mock()
    mock_client = mocker.Mock()
    mock_client.list.return_value = [faculty_experiment]
    mocker.patch("faculty.client", return_value=mock_client)

    mlflow_experiment = mocker.Mock()
    experiment_converter = mocker.patch(
        "mlflow_faculty.trackingstore.faculty_experiment_to_mlflow_experiment",
        return_value=mlflow_experiment,
    )
    lifecycle_stage = mocker.Mock()
    lifecycle_converter = mocker.patch(
        "mlflow_faculty.trackingstore"
        ".mlflow_viewtype_to_faculty_lifecycle_stage",
        return_value=lifecycle_stage,
    )

    store = FacultyRestStore(STORE_URI)
    experiments = store.list_experiments()

    mock_client.list.assert_called_once_with(PROJECT_ID, lifecycle_stage)
    experiment_converter.assert_called_once_with(faculty_experiment)
    lifecycle_converter.assert_called_once_with(ViewType.ACTIVE_ONLY)
    assert experiments == [mlflow_experiment]


def test_list_experiments_filtered_by_lifecycle_stage(mocker):
    faculty_experiment = mocker.Mock()
    mock_client = mocker.Mock()
    mock_client.list.return_value = [faculty_experiment]
    mocker.patch("faculty.client", return_value=mock_client)

    mlflow_experiment = mocker.Mock()
    experiment_converter = mocker.patch(
        "mlflow_faculty.trackingstore.faculty_experiment_to_mlflow_experiment",
        return_value=mlflow_experiment,
    )
    lifecycle_stage = mocker.Mock()
    lifecycle_converter = mocker.patch(
        "mlflow_faculty.trackingstore"
        ".mlflow_viewtype_to_faculty_lifecycle_stage",
        return_value=lifecycle_stage,
    )

    store = FacultyRestStore(STORE_URI)
    experiments = store.list_experiments(ViewType.DELETED_ONLY)

    mock_client.list.assert_called_once_with(PROJECT_ID, lifecycle_stage)
    experiment_converter.assert_called_once_with(faculty_experiment)
    lifecycle_converter.assert_called_once_with(ViewType.DELETED_ONLY)
    assert experiments == [mlflow_experiment]


def test_list_experiments_client_error(mocker):
    mock_client = mocker.Mock()
    mock_client.list.side_effect = HttpError(mocker.Mock(), "Error")
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(MlflowException, match="Error"):
        store.list_experiments()


def test_rename_experiment(mocker):
    mock_client = mocker.Mock()
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)
    store.rename_experiment(EXPERIMENT_ID, "new name")

    mock_client.update.assert_called_once_with(
        PROJECT_ID, EXPERIMENT_ID, name="new name"
    )


def test_rename_experiment_name_conflict(mocker):
    exception = ExperimentNameConflict("bad name")
    mock_client = mocker.Mock()
    mock_client.update.side_effect = exception
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(MlflowException, match=str(exception)):
        store.rename_experiment(EXPERIMENT_ID, "bad name")


def test_rename_experiment_client_error(mocker):
    mock_client = mocker.Mock()
    mock_client.update.side_effect = HttpError(mocker.Mock(), "Error")
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(MlflowException, match="Error"):
        store.rename_experiment(EXPERIMENT_ID, "new name")


def test_create_run(mocker):
    mlflow_timestamp = mocker.Mock()
    faculty_datetime = mocker.Mock()
    timestamp_converter_mock = mocker.patch(
        "mlflow_faculty.trackingstore."
        "mlflow_timestamp_to_datetime_milliseconds",
        return_value=faculty_datetime,
    )

    mlflow_tag = mocker.Mock()
    faculty_tag = mocker.Mock()
    tag_converter_mock = mocker.patch(
        "mlflow_faculty.trackingstore.mlflow_tag_to_faculty_tag",
        return_value=faculty_tag,
    )

    faculty_run = mocker.Mock()
    mock_client = mocker.Mock()
    mock_client.create_run.return_value = faculty_run
    mocker.patch("faculty.client", return_value=mock_client)

    mlflow_run = mocker.Mock()
    run_converter_mock = mocker.patch(
        "mlflow_faculty.trackingstore.faculty_run_to_mlflow_run",
        return_value=mlflow_run,
    )

    experiment_id = mocker.Mock()
    store = FacultyRestStore(STORE_URI)

    returned_run = store.create_run(
        experiment_id,
        "unused-mlflow-user-id",
        "unused-run-name",
        "unused-source-type",
        "unused-source-name",
        "unused-entry-point-name",
        mlflow_timestamp,
        "unused-source-version",
        [mlflow_tag],
        "unused-parent-run-id",
    )

    timestamp_converter_mock.assert_called_once_with(mlflow_timestamp)
    tag_converter_mock.assert_called_once_with(mlflow_tag)
    mock_client.create_run.assert_called_once_with(
        PROJECT_ID, experiment_id, faculty_datetime, tags=[faculty_tag]
    )
    run_converter_mock.assert_called_once_with(faculty_run)
    assert returned_run == mlflow_run


def test_create_run_client_error(mocker):
    mocker.patch(
        "mlflow_faculty.trackingstore."
        "mlflow_timestamp_to_datetime_milliseconds"
    )
    mocker.patch("mlflow_faculty.trackingstore.mlflow_tag_to_faculty_tag")

    mock_client = mocker.Mock()
    mock_client.create_run.side_effect = HttpError(mocker.Mock(), "Some error")
    mocker.patch("faculty.client", return_value=mock_client)

    experiment_id = mocker.Mock()
    mlflow_timestamp = mocker.Mock()
    mlflow_tag = mocker.Mock()

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(MlflowException, match="Some error"):
        store.create_run(
            experiment_id,
            "unused-mlflow-user-id",
            "unused-run-name",
            "unused-source-type",
            "unused-source-name",
            "unused-entry-point-name",
            mlflow_timestamp,
            "unused-source-version",
            [mlflow_tag],
            "unused-parent-run-id",
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
    run = store.get_run(RUN_UUID_HEX_STR)

    assert run == mock_mlflow_run

    mock_client.get_run.assert_called_once_with(PROJECT_ID, RUN_UUID)
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
        store.get_run(RUN_UUID_HEX_STR)


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
        store.search_runs([123], search_expressions=None, run_view_type=None)


def test_log_batch(mocker):
    mock_client = mocker.Mock()
    mocker.patch("faculty.client", return_value=mock_client)
    mlflow_metric = mocker.Mock()
    metric_converter_mock = mocker.patch(
        "mlflow_faculty.trackingstore.mlflow_metric_to_faculty_metric",
        return_value=mlflow_metric,
    )
    mlflow_param = mocker.Mock()
    param_converter_mock = mocker.patch(
        "mlflow_faculty.trackingstore.mlflow_param_to_faculty_param",
        return_value=mlflow_param,
    )
    mlflow_tag = mocker.Mock()
    tag_converter_mock = mocker.patch(
        "mlflow_faculty.trackingstore.mlflow_tag_to_faculty_tag",
        return_value=mlflow_tag,
    )

    store = FacultyRestStore(STORE_URI)
    store.log_batch(
        run_id=RUN_UUID_HEX_STR,
        metrics=[MLFLOW_METRIC],
        params=[MLFLOW_PARAM],
        tags=[MLFLOW_TAG],
    )

    mock_client.log_run_data.assert_called_once_with(
        PROJECT_ID,
        RUN_UUID,
        metrics=[mlflow_metric],
        params=[mlflow_param],
        tags=[mlflow_tag],
    )
    metric_converter_mock.assert_called_once_with(MLFLOW_METRIC)
    param_converter_mock.assert_called_once_with(MLFLOW_PARAM)
    tag_converter_mock.assert_called_once_with(MLFLOW_TAG)


def test_log_batch_empty(mocker):
    mock_client = mocker.Mock()
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)
    store.log_batch(RUN_UUID_HEX_STR)

    mock_client.log_run_data.assert_called_once_with(
        PROJECT_ID, RUN_UUID, metrics=[], params=[], tags=[]
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
            store.log_batch(RUN_UUID_HEX_STR)
    mock_client.log_run_data.assert_called_once_with(
        PROJECT_ID, RUN_UUID, metrics=[], params=[], tags=[]
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
            store.log_batch(RUN_UUID_HEX_STR)
    mock_client.log_run_data.assert_called_once_with(
        PROJECT_ID, RUN_UUID, metrics=[], params=[], tags=[]
    )


def test_log_metric(mocker):
    mock_client = mocker.Mock()
    mocker.patch("faculty.client", return_value=mock_client)
    mlflow_metric = mocker.Mock()
    metric_converter_mock = mocker.patch(
        "mlflow_faculty.trackingstore.mlflow_metric_to_faculty_metric",
        return_value=mlflow_metric,
    )
    store = FacultyRestStore(STORE_URI)
    store.log_metric(
        RUN_UUID_HEX_STR,
        MLFLOW_METRIC
    )
    mock_client.log_run_data.assert_called_once_with(
        PROJECT_ID,
        RUN_UUID,
        metrics=[mlflow_metric],
        params=[],
        tags=[]
    )
    metric_converter_mock.assert_called_once_with(MLFLOW_METRIC)


def test_delete_experiment(mocker):
    mock_client = mocker.Mock()
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)
    store.delete_experiment(EXPERIMENT_ID)

    mock_client.delete.assert_called_once_with(PROJECT_ID, EXPERIMENT_ID)


def test_delete_client_error(mocker):
    mock_client = mocker.Mock()
    mock_client.delete.side_effect = HttpError(
        mocker.Mock(), "Experiment with ID _ not found in project _"
    )
    mocker.patch("faculty.client", return_value=mock_client)

    store = FacultyRestStore(STORE_URI)

    with pytest.raises(
        MlflowException, match="Experiment with ID _ not found in project _"
    ):
        store.delete_experiment(EXPERIMENT_ID)

    mock_client.delete.assert_called_once_with(PROJECT_ID, EXPERIMENT_ID)
