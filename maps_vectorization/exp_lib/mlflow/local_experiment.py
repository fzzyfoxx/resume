import mlflow
import sys

experiment_name = sys.argv[1]

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_name)
mlflow.tensorflow.autolog(log_datasets=False, log_models=True, disable=True, checkpoint=False)