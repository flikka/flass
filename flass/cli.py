import logging
import sys
import os

from azureml.core import Workspace, Experiment
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget

import click
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.externals import joblib
from flass.model import train, get_data, plot_incorrect


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


@click.option("--plot/--no-plot", default=False)
@click.option("--batch-size", default=256)
@click.option("--epochs", default=3)
@click.option("--dataset", required=True, help="Choose between {fashion, mnist}")
@click.option("--model-type", required=False, default="kerasconv")
@click.option("--subset", required=False, default=-1)
@click.command()
def flass(plot, batch_size, epochs, dataset, model_type, subset):
    logger.info("Obtaining data")
    data, class_names = get_data(dataset, subset)

    azureml_experiment = setup_azureml(experiment_name=dataset)

    (x_train, y_train), (x_test, y_test) = data
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    with azureml_experiment.start_logging() as run:
        run.log("batch_size", batch_size)
        run.log("batch_size", batch_size)
        run.log("epochs", epochs)
        run.log("num_train_instances", len(x_train))
        run.log("ml_method", model_type)

        trained_pipeline = train(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            model_type=model_type,
        )

        if model_type != "kerasconv":
            joblib.dump(value=trained_pipeline, filename="model.pkl")

        predicted_y_probabilities = trained_pipeline.predict(x_test)
        roc_auc = roc_auc_score(y_test, predicted_y_probabilities, multi_class="ovr")
        run.log("AUC", roc_auc)
        if predicted_y_probabilities.shape[-1] > 1:
            y_predicted = predicted_y_probabilities.argmax(axis=-1)
        else:
            y_predicted = (predicted_y_probabilities > 0.5).astype("int32")

        matching_predictions = (y_predicted == y_test).tolist()
        correct = matching_predictions.count(True)
        incorrect = matching_predictions.count(False)
        run.log("correct_count", correct)
        run.log("incorrect_count", incorrect)

        report = classification_report(
            y_test, y_predicted, target_names=class_names, output_dict=True
        )
        for key in report.keys():
            if type(report[key]) == dict:
                for metric in report[key].keys():
                    run.log(f"{key}-{metric}", report[key][metric])
            else:
                run.log(f"{key}", report[key])

        if plot:
            plot_incorrect(x_test, y_test, y_predicted, class_names)


def setup_azureml(experiment_name: str):
    # Using a config that must be put in one of the places it will be found (eg. project root?)
    ws = Workspace.from_config()
    exp = Experiment(workspace=ws, name=experiment_name)
    setup_compute(ws)
    return exp


def setup_compute(ws):
    compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpucluster")
    compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
    compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 2)

    # This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")

    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            logger.info("found compute target. just use it. " + compute_name)
    else:
        logger.info("Creating a new compute target...")
        provisioning_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size, min_nodes=compute_min_nodes, max_nodes=compute_max_nodes
        )

        # create the cluster
        compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

        # can poll for a minimum number of nodes and for a specific timeout.
        # if no min node count is provided it will use the scale settings for the cluster
        compute_target.wait_for_completion(
            show_output=True, min_node_count=None, timeout_in_minutes=20
        )

        # For a more detailed view of current AmlCompute status, use get_status()
        logger.info(compute_target.get_status().serialize())


if __name__ == "__main__":
    flass()
