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
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
    RunStatus,
    RunTag,
    ViewType,
)
from mlflow_faculty.py23 import to_timestamp
from mlflow.exceptions import MlflowException
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_PARENT_RUN_ID

_FACULTY_TO_MLFLOW_RUN_STATUS_MAP = {
    FacultyExperimentRunStatus.RUNNING: "RUNNING",
    FacultyExperimentRunStatus.FINISHED: "FINISHED",
    FacultyExperimentRunStatus.FAILED: "FAILED",
    FacultyExperimentRunStatus.SCHEDULED: "SCHEDULED",
    FacultyExperimentRunStatus.KILLED: "KILLED",
}


# Below we are mapping both strings and RunStatus objects to Faculty RunStatus
# objects so we are less sensitive to future MLflow API changes
_MLFLOW_TO_FACULTY_RUN_STATUS_MAP = {
    "RUNNING": FacultyExperimentRunStatus.RUNNING,
    "FINISHED": FacultyExperimentRunStatus.FINISHED,
    "FAILED": FacultyExperimentRunStatus.FAILED,
    "SCHEDULED": FacultyExperimentRunStatus.SCHEDULED,
    "KILLED": FacultyExperimentRunStatus.KILLED,
    RunStatus.RUNNING: FacultyExperimentRunStatus.RUNNING,
    RunStatus.FINISHED: FacultyExperimentRunStatus.FINISHED,
    RunStatus.FAILED: FacultyExperimentRunStatus.FAILED,
    RunStatus.SCHEDULED: FacultyExperimentRunStatus.SCHEDULED,
    RunStatus.KILLED: FacultyExperimentRunStatus.KILLED,
}


_LIFECYCLE_STAGE_CONVERSION_MAP = {
    ViewType.ACTIVE_ONLY: FacultyLifecycleStage.ACTIVE,
    ViewType.DELETED_ONLY: FacultyLifecycleStage.DELETED,
    ViewType.ALL: None,
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

    tag_dict = {tag.key: tag.value for tag in faculty_run.tags}

    extra_mlflow_tags = []

    # Set run name tag if set as attribute but not already a tag
    if MLFLOW_RUN_NAME not in tag_dict and faculty_run.name:
        extra_mlflow_tags.append(RunTag(MLFLOW_RUN_NAME, faculty_run.name))

    # Set parent run ID tag if set as attribute but not already a tag
    if (
        MLFLOW_PARENT_RUN_ID not in tag_dict
        and faculty_run.parent_run_id is not None
    ):
        extra_mlflow_tags.append(
            RunTag(MLFLOW_PARENT_RUN_ID, faculty_run.parent_run_id.hex)
        )

    run_info = RunInfo(
        run_uuid=faculty_run.id.hex,
        experiment_id=faculty_run.experiment_id,
        user_id="",
        status=_FACULTY_TO_MLFLOW_RUN_STATUS_MAP[faculty_run.status],
        start_time=start_time,
        end_time=end_time,
        lifecycle_stage=lifecycle_stage,
        artifact_uri=faculty_run.artifact_location,
        run_id=faculty_run.id.hex,
    )
    run_data = RunData(
        params=[
            faculty_param_to_mlflow_param(param)
            for param in faculty_run.params
        ],
        metrics=[
            faculty_metric_to_mlflow_metric(metric)
            for metric in faculty_run.metrics
        ],
        tags=[faculty_tag_to_mlflow_tag(tag) for tag in faculty_run.tags]
        + extra_mlflow_tags,
    )
    run = Run(run_info, run_data)
    return run


def faculty_metric_to_mlflow_metric(faculty_metric):
    return Metric(
        key=faculty_metric.key,
        value=faculty_metric.value,
        timestamp=_datetime_to_mlflow_timestamp(faculty_metric.timestamp),
        step=faculty_metric.step,
    )


def mlflow_metric_to_faculty_metric(mlflow_metric):
    return FacultyMetric(
        key=mlflow_metric.key,
        value=mlflow_metric.value,
        timestamp=mlflow_timestamp_to_datetime(mlflow_metric.timestamp),
        step=mlflow_metric.step,
    )


def faculty_param_to_mlflow_param(faculty_param):
    return Param(key=faculty_param.key, value=faculty_param.value)


def mlflow_param_to_faculty_param(mlflow_param):
    return FacultyParam(key=mlflow_param.key, value=mlflow_param.value)


def mlflow_tag_to_faculty_tag(mlflow_tag):
    return FacultyTag(key=mlflow_tag.key, value=mlflow_tag.value)


def faculty_tag_to_mlflow_tag(faculty_tag):
    return RunTag(key=faculty_tag.key, value=faculty_tag.value)


def mlflow_timestamp_to_datetime(mlflow_timestamp):
    return datetime.fromtimestamp(mlflow_timestamp / 1000.0, tz=UTC)


def mlflow_viewtype_to_faculty_lifecycle_stage(mlflow_view_type):
    return _LIFECYCLE_STAGE_CONVERSION_MAP[mlflow_view_type]


def mlflow_to_faculty_run_status(mlflow_run_status):
    return _MLFLOW_TO_FACULTY_RUN_STATUS_MAP[mlflow_run_status]
