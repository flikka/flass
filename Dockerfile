FROM python:3.8

RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser
ENV PATH "/home/appuser/.local/bin:${PATH}"
RUN echo ${PATH}

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY flass/ ./flass
COPY setup.py .

RUN pwd
RUN ls -lah

RUN pip install .

