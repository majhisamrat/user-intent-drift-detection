FROM apache/airflow:2.8.1

# Switch to airflow user (IMPORTANT)
USER airflow

# Install ML dependencies as airflow user
COPY requirements-airflow.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
