# To enable ssh & remote debugging on app service change the base image to the one below
# FROM mcr.microsoft.com/azure-functions/python:2.0-python3.6-appservice
FROM mcr.microsoft.com/azure-functions/python:2.0-python3.6

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true

RUN python -m pip install --upgrade pip
RUN pip install azure-functions

COPY ./flass/ ./flass/
COPY ./setup.py .
RUN python setup.py install

COPY ./flass-az-docker-func /home/site/wwwroot
