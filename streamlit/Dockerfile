FROM python:3.7

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install lime
RUN pip install streamlit

RUN mkdir /app
WORKDIR /app

COPY setup.py .
COPY flass/ ./flass/
COPY streamlit/ ./streamlit/

RUN pip install .
RUN flass --dataset mnist

CMD streamlit run streamlit/flasslit.py -- $(cat mlflow_last_run.txt)
