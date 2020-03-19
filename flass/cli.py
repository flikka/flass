import logging
import sys

import click
import numpy as np
import mlflow
from sklearn.metrics import roc_auc_score, classification_report
from flass.model import train, get_data, plot_incorrect

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


@click.option("--plot/--no-plot", default=False)
@click.option("--batch-size", default=256)
@click.option("--epochs", default=3)
@click.option(
    "--dataset", default="fashion", required=True, help="Choose between \{fashion\}"
)
@click.command()
def flass(plot, batch_size, epochs, dataset):
    logger.info("Obtaining data")
    data, class_names = get_data(dataset)

    (x_train, y_train), (x_test, y_test) = data
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    with mlflow.start_run(run_name=dataset):
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("num_train_instances", len(x_train))
        trained_pipeline = train(x_train, y_train, batch_size=batch_size, epochs=epochs)

        predicted_y_probabilities = trained_pipeline.predict(x_test)
        roc_auc = roc_auc_score(y_test, predicted_y_probabilities, multi_class="ovr")
        mlflow.log_metric("AUC", roc_auc)
        if predicted_y_probabilities.shape[-1] > 1:
            y_predicted = predicted_y_probabilities.argmax(axis=-1)
        else:
            y_predicted = (predicted_y_probabilities > 0.5).astype("int32")

        report = classification_report(
            y_test, y_predicted, target_names=class_names, output_dict=True
        )
        for key in report.keys():
            if type(report[key]) == dict:
                for metric in report[key].keys():
                    mlflow.log_metric(f"{key}-{metric}", report[key][metric])
            else:
                mlflow.log_metric(f"{key}", report[key])

    if plot:
        plot_incorrect(x_test, y_test, y_predicted, class_names)


if __name__ == "__main__":
    flass()
