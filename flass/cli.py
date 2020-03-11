import logging
import sys

import click
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from flass.model import train, get_fashion_train_test, plot_incorrect

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


@click.option("--plot/--no-noplot", default=False)
@click.command()
def flass(plot):
    # Map for human readable class names
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    logger.info("Obtaining data")
    (x_train, y_train), (x_test, y_test) = get_fashion_train_test()
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    trained_pipeline = train(x_train, y_train, batch_size=1000)

    predicted_y_probabilities = trained_pipeline.predict(x_test)
    roc_auc = roc_auc_score(y_test, predicted_y_probabilities, multi_class="ovr")
    print(f"Area under ROC curve: {roc_auc}")

    if predicted_y_probabilities.shape[-1] > 1:
        y_predicted = predicted_y_probabilities.argmax(axis=-1)
    else:
        y_predicted = (predicted_y_probabilities > 0.5).astype("int32")

    print(classification_report(y_test, y_predicted, target_names=class_names))

    if plot:
        plot_incorrect(x_test, y_test, y_predicted, class_names)
