import os
import json
import logging

import azure.functions as func
from flass.model import load_mlflow_model
import flass
import numpy as np

CACHED_MODEL = None


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info(f"Flass is at {flass.__file__}")
    logging.info("Python HTTP trigger function processed a request to Flass func.")

    model_location = os.getenv("MLFLOW_MODEL_LOCATION")
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    connection_string_length = len(connection_string) if connection_string else 0

    response = f"Welcome to Flass. Model location is {model_location}\n"
    response += (
        f"Lenght of model storage connection string is {connection_string_length}\n"
    )
    global CACHED_MODEL
    if CACHED_MODEL:
        logging.info("Model found in cache, using this")
        trained_model = CACHED_MODEL
    else:
        logging.info(f"Model not found in cache, loading from {model_location}")
        trained_model = load_mlflow_model(model_location)
        CACHED_MODEL = trained_model
        logging.info(f"Model loaded successfully")

    payload_from_param = req.params.get("payload")
    if payload_from_param:
        logging.info(
            f"Trying to param payload of type {type(payload_from_param)} "
            f"and value {payload_from_param} into json"
        )
        payload = json.loads(payload_from_param)

    else:
        try:
            req_body = req.get_json()
        except ValueError:
            payload = None
        else:
            payload = req_body.get("payload")

    if payload:
        logging.info(
            f"Trying to load payload of type {type(payload)} "
            f"and value {payload} into numpy array"
        )
        try:
            np_array = np.array(payload)
        except Exception as e:
            logging.error(f"Failed to parse, error: {e}")

        logging.info(f"Arryay loaded, dimensions {np_array.shape}")
        result = trained_model.predict(payload)

        return func.HttpResponse(json.dumps(result.tolist()))
    else:
        return func.HttpResponse(
            response + "\nPlease pass data in the key 'payload' on the query string "
            "or in the request body",
            status_code=400,
        )
