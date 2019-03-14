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
    )
    run_data = RunData(tags=mlflow_tags_to_faculty_tags(faculty_run.tags))
    run = Run(run_info, run_data)
    return run


def mlflow_metrics_to_faculty_metrics(mlflow_metrics):
    return [
        FacultyMetric(
            key=metric.key,
            value=metric.value,
            timestamp=mlflow_timestamp_to_datetime_seconds(metric.timestamp),
        )
        for metric in mlflow_metrics
    ]


def mlflow_params_to_faculty_params(mlflow_params):
    return [
        FacultyParam(key=param.key, value=param.value)
        for param in mlflow_params
    ]


def mlflow_tags_to_faculty_tags(mlflow_tags):
    return [FacultyTag(key=tag.key, value=tag.value) for tag in mlflow_tags]


def mlflow_timestamp_to_datetime_milliseconds(mlflow_timestamp):
    return datetime.fromtimestamp(mlflow_timestamp / 1000.0, tz=UTC)


def mlflow_timestamp_to_datetime_seconds(mlflow_timestamp):
    return datetime.fromtimestamp(mlflow_timestamp, tz=UTC)
