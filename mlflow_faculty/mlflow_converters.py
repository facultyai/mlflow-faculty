from datetime import datetime

from pytz import UTC

from faculty.clients.experiment import (
    ExperimentRunStatus as FacultyExperimentRunStatus,
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
            faculty_http_error.response.text,
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
    )
    run_data = RunData()
    run = Run(run_info, run_data)
    return run


def mlflow_run_metric_to_faculty_run_metric(mlflow_run_metric):
    return FacultyMetric(
        key=mlflow_run_metric.key,
        value=mlflow_run_metric.value,
        timestamp=mlflow_timestamp_to_datetime_milliseconds(
            mlflow_run_metric.timestamp
        ),
    )


def mlflow_run_param_to_faculty_run_param(mlflow_run_param):
    return FacultyParam(key=mlflow_run_param.key, value=mlflow_run_param.value)


def mlflow_run_tag_to_faculty_run_tag(mlflow_run_tag):
    return FacultyTag(
        key=mlflow_run_tag.key,
        value=mlflow_run_tag.value,
        timestamp=mlflow_timestamp_to_datetime_milliseconds(
            mlflow_run_tag.timestamp
        ),
    )


def mlflow_timestamp_to_datetime_milliseconds(mlflow_timestamp):
    return datetime.fromtimestamp(mlflow_timestamp / 1000.0, tz=UTC)


def mlflow_timestamp_to_datetime_seconds(mlflow_timestamp):
    return datetime.fromtimestamp(mlflow_timestamp, tz=UTC)
