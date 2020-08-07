import argparse
import json
import logging
import os.path

import mlflow
import numpy as np
from PIL import Image
import streamlit as st

from flass.model import load_mlflow_model


def make_sidebar(mlflow_run, class_names):
    st.title("Flass image classifier")
    st.sidebar.subheader("Information about model")
    st.sidebar.markdown(f"Model type:\t{mlflow_run.data.params.get('ml_method')}")
    st.sidebar.markdown(f"Dataset used: {mlflow_run.data.params.get('dataset')}")
    st.sidebar.markdown(
        f"Train size:\t{mlflow_run.data.params.get('num_train_instances')}"
    )

    st.sidebar.markdown(f"Class names: ```{class_names}```")

    st.sidebar.subheader("Upload an image to classify. It will be resized")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file to classify with model", type=["png", "jpg", "jpeg"]
    )
    return uploaded_file


def flasslit(mlflowrun: str):
    st.set_option('deprecation.showfileUploaderEncoding', False)
    mlflow_run = mlflow.get_run(mlflowrun)
    class_names = json.loads(mlflow_run.data.params.get("class_names"))
    modelpath = os.path.join(mlflow_run.info.artifact_uri, "saved-model")

    uploaded_file = make_sidebar(mlflow_run, class_names)

    image_array = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.subheader("Uploaded image information")
        st.write(image)
        st.image(image)
        image = image.resize((28, 28), Image.ANTIALIAS).convert('L')
        st.subheader("Resized image information")
        st.write(image)
        st.image(image)
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, -1)

    logging.info(f"Loading model from {modelpath}")
    loaded_model = load_mlflow_model(modelpath)
    if image_array is not None:
        res = loaded_model.predict(np.array([image_array]))[0]
        res = [float(prob) for prob in res]
        st.subheader("Predicted class probabilities")
        st.write(dict(zip(class_names, res)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments to Flass Streamlit app")
    parser.add_argument("mlflowrun", type=str, help="MLFLow Run to fetch info from")
    args = parser.parse_args()

    flasslit(args.mlflowrun)
