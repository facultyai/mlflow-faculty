from mlflow.entities import Experiment, LifecycleStage, RunInfo, RunStatus


def faculty_experiment_to_mlflow_experiment(faculty_experiment):
    active = faculty_experiment.deleted_at is None
    return Experiment(
        faculty_experiment.id,
        faculty_experiment.name,
        faculty_experiment.artifact_location,
        LifecycleStage.ACTIVE if active else LifecycleStage.DELETED,
    )


def faculty_run_to_mlflow_run(faculty_run):
    run_info = RunInfo(
        faculty_run.id,
        faculty_run.experiment_id,
        "",  # name
        "",  # source_type
        "",  # source_name
        "",  # entry_point_name
        "",  # user_id
        RunStatus.RUNNING,
        faculty_run.started_at,
        faculty_run.ended_at,
        "",  # shource version
        LifecycleStage.ACTIVE
    )
    return run_info
