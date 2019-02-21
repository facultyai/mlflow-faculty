from mlflow.entities import Experiment, LifecycleStage


def faculty_experiment_to_mlflow_experiment(faculty_experiment):
    active = faculty_experiment.deleted_at is None
    return Experiment(
        faculty_experiment.id,
        faculty_experiment.name,
        faculty_experiment.artifact_location,
        LifecycleStage.ACTIVE if active else LifecycleStage.DELETED,
    )
