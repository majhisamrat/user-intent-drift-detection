from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "samrat",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="intent_classification_ml_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["nlp", "mlops", "intent"]
) as dag:

    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command="python /opt/airflow/src/preprocess.py"
    )

    vectorize = BashOperator(
        task_id="vectorize_text",
        bash_command="python /opt/airflow/src/vectorize.py"
    )

    train = BashOperator(
        task_id="train_model",
        bash_command="python /opt/airflow/src/train.py"
    )

    create_baseline = BashOperator(
        task_id="create_baseline_distribution",
        bash_command="python /opt/airflow/src/create_baseline.py"
    )

    evaluate = BashOperator(
        task_id="evaluate_model",
        bash_command="python /opt/airflow/src/confusion_analysis.py"
    )

    detect_intent_drift = BashOperator(
        task_id="detect_intent_drift",
        bash_command="python /opt/airflow/src/detect_drift.py"
    )

    detect_confidence_drift = BashOperator(
        task_id="detect_confidence_drift",
        bash_command="python /opt/airflow/src/confidence_drift.py"
    )

    (
        preprocess
        >> vectorize
        >> train
        >> create_baseline
        >> evaluate
        >> detect_intent_drift
        >> detect_confidence_drift
    )
