from datetime import datetime

from pytz import UTC

from mlflow.entities import (
    Experiment,
    LifecycleStage,
    RunInfo,
    RunStatus,
    Run,
    RunData,
)
from faculty.clients.experiment import (
    ExperimentRunStatus as FacultyExperimentRunStatus,
)

from mlflow_faculty.py23 import to_timestamp


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
        faculty_run.id,
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


def mlflow_timestamp_to_datetime(mlflow_timestamp):
    return datetime.fromtimestamp(mlflow_timestamp / 1000, tz=UTC)
