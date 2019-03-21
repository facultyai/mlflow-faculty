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

from datetime import datetime

from pytz import UTC

from faculty.clients.experiment import (
    ExperimentRunStatus as FacultyExperimentRunStatus,
    LifecycleStage as FacultyLifecycleStage,
    Metric as FacultyMetric,
    Param as FacultyParam,
    Tag as FacultyTag,
)
from mlflow.entities import (
    Experiment,
    LifecycleStage,
    Run,
    RunData,
    RunInfo,
    RunStatus,
    RunTag,
    ViewType,
)
from mlflow_faculty.py23 import to_timestamp
from mlflow.exceptions import MlflowException

_RUN_STATUS_MAP = {
    FacultyExperimentRunStatus.RUNNING: RunStatus.RUNNING,
    FacultyExperimentRunStatus.FINISHED: RunStatus.FINISHED,
    FacultyExperimentRunStatus.FAILED: RunStatus.FAILED,
    FacultyExperimentRunStatus.SCHEDULED: RunStatus.SCHEDULED,
}


def _datetime_to_mlflow_timestamp(dt):
    return to_timestamp(dt) * 1000


def faculty_experiment_to_mlflow_experiment(faculty_experiment):
    active = faculty_experiment.deleted_at is None
    return Experiment(
        faculty_experiment.id,
        faculty_experiment.name,
        faculty_experiment.artifact_location,
        LifecycleStage.ACTIVE if active else LifecycleStage.DELETED,
    )


def faculty_http_error_to_mlflow_exception(faculty_http_error):
    return MlflowException(
        "{}. Received response {} with status code {}".format(
            faculty_http_error.error,
            faculty_http_error.error_code,
            faculty_http_error.response.status_code,
        )
    )


def faculty_run_to_mlflow_run(faculty_run):
    lifecycle_stage = (
        LifecycleStage.ACTIVE
        if faculty_run.deleted_at is None
        else LifecycleStage.DELETED
    )
    start_time = _datetime_to_mlflow_timestamp(faculty_run.started_at)
    end_time = (
        _datetime_to_mlflow_timestamp(faculty_run.ended_at)
        if faculty_run.ended_at is not None
        else None
    )
    run_info = RunInfo(
        faculty_run.id.hex,
        faculty_run.experiment_id,
        "",  # name
        "",  # source_type
        "",  # source_name
        "",  # entry_point_name
        "",  # user_id
        _RUN_STATUS_MAP[faculty_run.status],
        start_time,
        end_time,
        "",  # source version
        lifecycle_stage,
        faculty_run.artifact_location,
    )
    run_data = RunData(
        tags=[faculty_tag_to_mlflow_tag(tag) for tag in faculty_run.tags]
    )
    run = Run(run_info, run_data)
    return run


def mlflow_metric_to_faculty_metric(mlflow_metric):
    return FacultyMetric(
        key=mlflow_metric.key,
        value=mlflow_metric.value,
        timestamp=mlflow_timestamp_to_datetime_seconds(
            mlflow_metric.timestamp
        ),
    )


def mlflow_param_to_faculty_param(mlflow_param):
    return FacultyParam(key=mlflow_param.key, value=mlflow_param.value)


def mlflow_tag_to_faculty_tag(mlflow_tag):
    return FacultyTag(key=mlflow_tag.key, value=mlflow_tag.value)


def faculty_tag_to_mlflow_tag(faculty_tag):
    return RunTag(key=faculty_tag.key, value=faculty_tag.value)


def mlflow_timestamp_to_datetime_milliseconds(mlflow_timestamp):
    return datetime.fromtimestamp(mlflow_timestamp / 1000.0, tz=UTC)


def mlflow_timestamp_to_datetime_seconds(mlflow_timestamp):
    return datetime.fromtimestamp(mlflow_timestamp, tz=UTC)


def mlflow_viewtype_to_faculty_lifecycle_stage(mlflow_view_type):
    if mlflow_view_type == ViewType.ACTIVE_ONLY:
        return FacultyLifecycleStage.ACTIVE
    elif mlflow_view_type == ViewType.DELETED_ONLY:
        return FacultyLifecycleStage.DELETED
    elif mlflow_view_type == ViewType.ALL:
        return None
    raise ValueError("Unexpected view_type: {}".format(mlflow_view_type))
