import json
import logging
from pprint import pformat
import random
import sys
import click

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import mlflow.keras
import mlflow.exceptions

import numpy as np

from skimage.segmentation import mark_boundaries
from sklearn.metrics import roc_auc_score, classification_report
from flass.model import train, get_data, plot_incorrect

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)-2s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm

except ImportError:
    logger.warning("Dependency lime not found, please install with [lime] to enable")


@click.option("--plot/--no-plot", default=False)
@click.option("--batch-size", default=256)
@click.option("--epochs", default=3)
@click.option("--dataset", required=True, help="Choose between {fashion, mnist}")
@click.option("--model-type", required=False, default="kerasconv")
@click.option("--subset", required=False, default=-1)
@click.option("--lime/--no-lime", default=False)
@click.option("--run-name", required=False)
@click.command()
def flass(plot, batch_size, epochs, dataset, model_type, subset, lime, run_name):
    logger.info("Obtaining data")
    data, class_names = get_data(dataset, subset)

    (x_train, y_train), (x_test, y_test) = data

    with mlflow.start_run(run_name=run_name):
        logger.info(f"Artifact URI: {mlflow.get_artifact_uri()}")
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("num_train_instances", len(x_train))
        mlflow.log_param("ml_method", model_type)
        trained_model = train(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            model_type=model_type,
        )

        trained_model.log_model(trained_model, "saved-model")
        model_location = mlflow.get_artifact_uri("saved-model")
        logger.info(f"Loading model from: {model_location}")

        # The pyfunc flavour of Keras seems to require a Pandas dataframe, but the
        # Keras flavour (as used when using mlflow.keras) appears to accept the higher
        # dimensioned numpy input
        try:
            loaded_model = mlflow.keras.load_model(model_location)
            logger.info("Loaded MLFlow model with keras flavour")
        except mlflow.exceptions.MlflowException:
            loaded_model = mlflow.pyfunc.load_model(model_location)
            logger.info("Loaded MLFlow model with pyfunc flavour")

        if lime:
            # Do a LIME
            samples = random.sample(range(0, len(x_test)), 10)
            for i in samples:
                limeify(x_test[i], loaded_model, class_names)

        predicted_y_probabilities = loaded_model.predict(x_test)
        roc_auc = roc_auc_score(y_test, predicted_y_probabilities, multi_class="ovr")
        mlflow.log_metric("AUC", roc_auc)
        logger.info(f"AUC: {roc_auc}")
        if predicted_y_probabilities.shape[-1] > 1:
            y_predicted = predicted_y_probabilities.argmax(axis=-1)
        else:
            y_predicted = (predicted_y_probabilities > 0.5).astype("int32")

        matching_predictions = (y_predicted == y_test).tolist()
        correct = matching_predictions.count(True)
        incorrect = matching_predictions.count(False)
        mlflow.log_metric("correct_count", correct)
        logger.info(f"Correct predictions: {correct}")
        mlflow.log_metric("incorrect_count", incorrect)
        logger.info(f"Incorrect classifications: {incorrect}")
        logger.info("See MLFlow for more details from run")

        report = classification_report(
            y_test, y_predicted, target_names=class_names, output_dict=True
        )
        classification_report_file = "/tmp/classification_report.json"
        with open(classification_report_file, "w") as file:
            json.dump(report, file)
        logger.info(
            f'Wrote classification report to file "{classification_report_file}"'
        )
        for key in report.keys():
            if type(report[key]) == dict:
                for metric in report[key].keys():
                    mlflow.log_metric(f"{key}-{metric}", report[key][metric])
            else:
                mlflow.log_metric(f"{key}", report[key])

    if plot:
        plot_incorrect(x_test, y_test, y_predicted, class_names)


def limeify(image_to_explain, trained_pipeline, class_names):
    logger.info("Start a LIME explanation")
    lime_image_probabilities = trained_pipeline.predict(np.array([image_to_explain]))[0]
    image_probabilities = tuple(zip(class_names, lime_image_probabilities))
    logger.info(
        "Models predicted probabilities for image:\n" + pformat(image_probabilities)
    )
    plt.imshow(image_to_explain)
    plt.show()
    explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm(
        "quickshift", kernel_size=1, max_dist=200, ratio=0.2
    )
    explanation = explainer.explain_instance(
        image_to_explain,
        trained_pipeline.predict,
        top_labels=10,
        num_samples=1000,
        segmentation_fn=segmenter,
    )
    logger.info("Done with a LIME")
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False
    )
    plt.imshow(mark_boundaries(temp, mask))
    plt.show()


if __name__ == "__main__":
    flass()
