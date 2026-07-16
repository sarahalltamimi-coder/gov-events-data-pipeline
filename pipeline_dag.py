# airflow dag that runs the whole pipeline in order:
# ingestion -> lakehouse -> quality gate -> rag

from datetime import datetime, timedelta

from loguru import logger

import ingestion
import lakehouse
import quality_check
import rag_pipeline

# save the logs also to a file
logger.add("pipeline.log", level="DEBUG")


def run_quality():
    passed = quality_check.main()
    if not passed:
        # if the quality gate fails we stop the pipeline here
        raise Exception("quality gate failed, stopping the pipeline")


# the tasks in order
TASKS = [
    ("ingest", ingestion.main),
    ("lakehouse", lakehouse.main),
    ("quality_gate", run_quality),
    ("rag", rag_pipeline.main),
]

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    HAS_AIRFLOW = True

    with DAG(
        dag_id="events_pipeline",
        schedule="@daily",
        start_date=datetime(2026, 7, 1),
        catchup=False,
        default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    ) as dag:
        ops = [PythonOperator(task_id=name, python_callable=fn) for name, fn in TASKS]
        # ingest >> lakehouse >> quality_gate >> rag
        for a, b in zip(ops, ops[1:]):
            a >> b

except ImportError:
    HAS_AIRFLOW = False
    dag = None


def run_without_airflow():
    for name, fn in TASKS:
        print(f"\n########## task: {name} ##########")
        logger.info(f"starting task {name}")
        fn()
        logger.success(f"task {name} done")


def main():
    print("---- orchestration ----")
    if HAS_AIRFLOW:
        logger.info("airflow found, running the dag with dag.test()")
        dag.test()
    else:
        logger.warning("airflow not installed, running the tasks in order manually")
        run_without_airflow()


if __name__ == "__main__":
    main()
