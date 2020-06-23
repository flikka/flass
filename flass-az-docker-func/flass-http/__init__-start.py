import os
import logging


import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request to Flass func.")
    model_location = os.getenv("MLFLOW_MODEL_LOCATION")
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    connection_string_length = len(connection_string) if connection_string else 0
    payload = req.params.get("payload")
    if not payload:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            payload = req_body.get("payload")

    if payload:
        response = f"Welcome to Flass. Model location is {model_location}\n"
        response += (
            f"Lenght of model storage connection string is {connection_string_length}\n"
        )
        response += f"For the Flassprediction I got this input:\n {payload}\n"
        return func.HttpResponse(response)
    else:
        return func.HttpResponse(
            "Please pass data in the key 'payload' on the query string "
            "or in the request body",
            status_code=400,
        )
