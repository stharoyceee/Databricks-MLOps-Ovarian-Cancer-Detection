import mlflow
from databricks.sdk.runtime import dbutils

DATASET_DIR = "/Volumes/compvis_od/data/ovarian_dataset/Training_Images/"
TISSUE_THRESHOLD = 0.50

VALIDATION_SIZE = 0.15

BATCH_SIZE = 16

VARIANCE_CUTOFF = 0.01
CORRELATION_THRESHOLD = 0.90
K_BEST_FEATURES = 200

VOLUME_FEATURES_DIR = "/Volumes/compvis_od/data/ovarian_dataset/processed_features/"

EXPERIMENT_NAME = "/Users/shrestharoyce24@gmail.com/Databricks MLOps and MLFlow with Computer Vision Models/ovarian_cancer_pipeline"
RUN_NAME = "ovarian_pipeline_run"

def get_run_id(experiment_name = EXPERIMENT_NAME, run_name = RUN_NAME):
    _experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if _experiment is None:
        raise ValueError(f"Experiment not found")

    _fallback_runs = mlflow.search_runs(
        experiment_ids=[_experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if _fallback_runs.empty:
        raise ValueError(f"No Run name found in the experiment")

      # dbutils is not auto-injected into imported workspace files

    run_id = dbutils.jobs.taskValues.get(
        taskKey="Data_Cleaning",
        key="mlflow_run_id",
        debugValue=_fallback_runs.iloc[0]["run_id"],
    )

    return run_id