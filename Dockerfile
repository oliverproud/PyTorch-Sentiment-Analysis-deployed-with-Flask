# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
# Copy application dependency manifests to the container image.
# Copying this separately prevents re-running pip install on every code change.

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN python model.py

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080",  "--workers", "1", "--threads", "8", "app:app", "--timeout", "900"]
