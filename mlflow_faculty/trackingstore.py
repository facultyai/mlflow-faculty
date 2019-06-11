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

from uuid import UUID

from six.moves import urllib

import faculty
import faculty.clients.base
import faculty.clients.experiment
from faculty.clients.experiment import (
    ParamConflict as FacultyParamConflict,
    ExperimentDeleted as FacultyExperimentDeleted,
)
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.store.abstract_store import AbstractStore
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_PARENT_RUN_ID

from mlflow_faculty.mlflow_converters import (
    faculty_experiment_to_mlflow_experiment,
    faculty_http_error_to_mlflow_exception,
    faculty_metric_to_mlflow_metric,
    faculty_run_to_mlflow_run,
    mlflow_timestamp_to_datetime,
    mlflow_metric_to_faculty_metric,
    mlflow_param_to_faculty_param,
    mlflow_tag_to_faculty_tag,
    mlflow_viewtype_to_faculty_lifecycle_stage,
    mlflow_to_faculty_run_status,
)


class FacultyRestStore(AbstractStore):
    def __init__(self, store_uri, **_):
        parsed_uri = urllib.parse.urlparse(store_uri)
        if parsed_uri.scheme != "faculty":
            raise ValueError("Not a faculty URI: {}".format(store_uri))
        # Test for PROJECT_ID in netloc rather than path.
        elif parsed_uri.netloc != "":
            raise ValueError(
                "Invalid URI {}. Netloc is reserved. "
                "Did you mean 'faculty:{}".format(store_uri, parsed_uri.netloc)
            )

        cleaned_path = parsed_uri.path.strip("/")
        try:
            self._project_id = UUID(cleaned_path)
        except ValueError:
            raise ValueError(
                "{} in given URI {} is not a valid UUID".format(
                    cleaned_path, store_uri
                )
            )

        self._client = faculty.client("experiment")

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        """
        :param view_type: Qualify requested type of experiments.

        :return: a list of Experiment objects stored in store for requested
            view.
        """
        lifecycle_stage = mlflow_viewtype_to_faculty_lifecycle_stage(view_type)
        try:
            faculty_experiments = self._client.list(
                self._project_id, lifecycle_stage
            )
        except faculty.clients.base.HttpError as e:
            raise faculty_http_error_to_mlflow_exception(e)
        else:
            return [
                faculty_experiment_to_mlflow_experiment(faculty_experiment)
                for faculty_experiment in faculty_experiments
            ]

    def create_experiment(self, name, artifact_location):
        """
        Creates a new experiment.
        If an experiment with the given name already exists, throws exception.

        :param name: Desired name for an experiment
        :param artifact_location: Base location for artifacts in runs. May be
            None.

        :return: experiment_id (integer) for the newly created experiment.
        """
        try:
            faculty_experiment = self._client.create(
                self._project_id, name, artifact_location=artifact_location
            )
        except faculty.clients.experiment.ExperimentNameConflict as e:
            raise MlflowException(str(e))
        except faculty.clients.base.HttpError as e:
            raise faculty_http_error_to_mlflow_exception(e)
        else:
            return faculty_experiment.id

    def get_experiment(self, experiment_id):
        """
        Fetches the experiment by ID from the backend store.

        :param experiment_id: Integer id for the experiment

        :return: A single :py:class:`mlflow.entities.Experiment` object if it
            exists, otherwise raises an exception.
        """
        try:
            faculty_experiment = self._client.get(
                self._project_id, int(experiment_id)
            )
        except faculty.clients.base.HttpError as e:
            raise faculty_http_error_to_mlflow_exception(e)
        else:
            return faculty_experiment_to_mlflow_experiment(faculty_experiment)

    def get_experiment_by_name(self, experiment_name):
        """
        Fetches the experiment by name from the backend store.

        :param experiment_name: Name of experiment

        :return: A single :py:class:`mlflow.entities.Experiment` object if it
            exists.
        """
        return super(FacultyRestStore, self).get_experiment_by_name(
            experiment_name
        )

    def delete_experiment(self, experiment_id):
        """
        Deletes the experiment from the backend store. Deleted experiments can
        be restored until permanently deleted.

        :param experiment_id: Integer id for the experiment
        """
        try:
            self._client.delete(self._project_id, int(experiment_id))
        except faculty.clients.base.HttpError as e:
            raise faculty_http_error_to_mlflow_exception(e)

    def restore_experiment(self, experiment_id):
        """
        Restore deleted experiment unless it is permanently deleted.

        :param experiment_id: Integer id for the experiment
        """
        try:
            self._client.restore(self._project_id, int(experiment_id))
        except faculty.clients.base.HttpError as e:
            raise faculty_http_error_to_mlflow_exception(e)

    def rename_experiment(self, experiment_id, new_name):
        """
        Update an experiment's name. The new name must be unique.

        :param experiment_id: Integer id for the experiment
        """
        try:
            self._client.update(
                self._project_id, int(experiment_id), name=new_name
            )
        except faculty.clients.experiment.ExperimentNameConflict as e:
            raise MlflowException(str(e))
        except faculty.clients.base.HttpError as e:
            raise faculty_http_error_to_mlflow_exception(e)

    def get_run(self, run_id):
        """
        Fetches the run from backend store

        :param run_id: string containing run UUID
            (32 hex characters = a uuid4 stripped off of dashes)

        :return: A single :py:class:`mlflow.entities.Run` object if it exists,
            otherwise raises an exception
        """
        try:
            faculty_run = self._client.get_run(self._project_id, UUID(run_id))
        except faculty.clients.base.HttpError as e:
            raise faculty_http_error_to_mlflow_exception(e)
        else:
            mlflow_run = faculty_run_to_mlflow_run(faculty_run)
            return mlflow_run

    def update_run_info(self, run_id, run_status, end_time):
        """
        Updates the metadata of the specified run.

        :param run_id: string containing run UUID
        :param run_status: RunStatus to update the run to, optional
        :param end_time: timestamp to update the run ended_at to, optional

        :return: :py:class:`mlflow.entities.RunInfo` describing the updated
            run.
        """
        try:
            faculty_run = self._client.update_run_info(
                self._project_id,
                UUID(run_id),
                mlflow_to_faculty_run_status(run_status),
                mlflow_timestamp_to_datetime(end_time),
            )
        except faculty.clients.base.HttpError as e:
            raise faculty_http_error_to_mlflow_exception(e)
        else:
            mlflow_run = faculty_run_to_mlflow_run(faculty_run)
            return mlflow_run.info

    def create_run(self, experiment_id, user_id, start_time, tags):
        """
        Creates a run under the specified experiment ID, setting the run's
        status to "RUNNING" and the start time to the current time.

        :param experiment_id: ID of the experiment for this run
        :param user_id: ID of the user launching this run
        :param start_time: Time the run started at in epoch milliseconds.
        :param tags: List of Mlflow Tag entities.

        :return: The created Run object
        """
        tags = [] if tags is None else tags

        # For backward compatability, fall back to run name or parent run ID
        # set in tags
        tag_dict = {tag.key: tag.value for tag in tags}
        run_name = tag_dict.get(MLFLOW_RUN_NAME) or ""
        parent_run_id = tag_dict.get(MLFLOW_PARENT_RUN_ID) or None

        try:
            faculty_run = self._client.create_run(
                self._project_id,
                int(experiment_id),
                run_name,
                mlflow_timestamp_to_datetime(start_time),
                None if parent_run_id is None else UUID(parent_run_id),
                tags=[mlflow_tag_to_faculty_tag(tag) for tag in tags],
            )
        except FacultyExperimentDeleted as conflict:
            raise MlflowException(
                "Experiment {0} is deleted."
                " To create runs for this experiment,"
                " first restore it with the shell command "
                "'mlflow experiments restore {0}'".format(
                    conflict.experiment_id
                )
            )
        except faculty.clients.base.HttpError as e:
            raise faculty_http_error_to_mlflow_exception(e)
        else:
            mlflow_run = faculty_run_to_mlflow_run(faculty_run)
            return mlflow_run

    def delete_run(self, run_id):
        """
        Deletes a run.
        :param run_id:
        """

        run_id = UUID(run_id)

        try:
            response = self._client.delete_runs(self._project_id, [run_id])
        except faculty.clients.base.HttpError as e:
            raise faculty_http_error_to_mlflow_exception(e)

        if run_id in response.deleted_run_ids:
            return
        elif run_id in response.conflicted_run_ids:
            raise MlflowException(
                "Could not delete already-deleted run {}".format(run_id.hex)
            )
        else:
            raise MlflowException(
                "Could not delete non-existent run {}".format(run_id.hex)
            )

    def restore_run(self, run_id):
        """
        Restores a run.
        :param run_id:
        """

        run_id = UUID(run_id)

        try:
            response = self._client.restore_runs(self._project_id, [run_id])
        except faculty.clients.base.HttpError as e:
            raise faculty_http_error_to_mlflow_exception(e)

        if run_id in response.restored_run_ids:
            return
        elif run_id in response.conflicted_run_ids:
            raise MlflowException(
                "Could not restore already-active run {}".format(run_id.hex)
            )
        else:
            raise MlflowException(
                "Could not restore non-existent run {}".format(run_id.hex)
            )

    def get_metric_history(self, run_id, metric_key):
        """
        Returns all logged value for a given metric.

        :param run_id: Unique identifier for run
        :param metric_key: Metric name within the run

        :return: A list of float values logged for the give metric if logged,
            else empty list
        """
        try:
            metric_history = self._client.get_metric_history(
                self._project_id, UUID(run_id), metric_key
            )
        except faculty.clients.base.HttpError as e:
            raise faculty_http_error_to_mlflow_exception(e)
        else:
            return [
                faculty_metric_to_mlflow_metric(faculty_metric)
                for faculty_metric in metric_history
            ]

    def search_runs(self, experiment_ids, search_expressions, run_view_type):
        """ Returns runs that match the given list of search expressions within
        the experiments.  Given multiple search expressions, all these
        expressions are ANDed together for search.

        :param experiment_ids: List[int] of experiment ids to scope the search
        :param search_expression: List of search expressions

        :return: A list of :py:class:`mlflow.entities.Run` objects that satisfy
            the search expressions
        """
        if search_expressions is not None:
            raise NotImplementedError("search_expressions must be set to None")
        if run_view_type is not None:
            raise NotImplementedError("run_view_type must be set to None")

        try:
            faculty_runs = []
            experiment_ids = (
                None
                if experiment_ids is None
                else list(map(int, experiment_ids))
            )
            while True:
                list_runs_response = self._client.list_runs(
                    self._project_id, experiment_ids=experiment_ids
                )
                faculty_runs.extend(list_runs_response.runs)
                if (
                    list_runs_response.pagination.next is None
                    or len(list_runs_response.runs) == 0
                ):
                    break
        except faculty.clients.base.HttpError as e:
            raise faculty_http_error_to_mlflow_exception(e)
        else:
            return [faculty_run_to_mlflow_run(run) for run in faculty_runs]

    def log_batch(self, run_id, metrics=None, params=None, tags=None):
        """
        Fetches the experiment by ID from the backend store.

        :param run_id: string containing run UUID
            (32 hex characters = a uuid4 stripped off of dashes)
        :param metrics: List of Mlflow Metric entities.
        :param params: List of Mlflow Param entities
        :param tags: List of Mlflow Tag entities.
        """
        metrics = [] if metrics is None else metrics
        params = [] if params is None else params
        tags = [] if tags is None else tags

        try:
            self._client.log_run_data(
                self._project_id,
                UUID(run_id),
                params=[
                    mlflow_param_to_faculty_param(param) for param in params
                ],
                metrics=[
                    mlflow_metric_to_faculty_metric(metric)
                    for metric in metrics
                ],
                tags=[mlflow_tag_to_faculty_tag(tag) for tag in tags],
            )
        except FacultyParamConflict as conflict:
            raise MlflowException(
                "Conflicting param keys: {}".format(
                    conflict.conflicting_params
                )
            )
        except faculty.clients.base.HttpError as e:
            raise faculty_http_error_to_mlflow_exception(e)
