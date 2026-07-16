# quality gate: great expectations checks + openlineage events

import os
import re
from datetime import datetime, UTC

import pandas as pd
from loguru import logger

CSV_FILE = "Public_Sector_Events_Q2_2026_CSV.csv"

COLS = {
    "اسم الجهة الحكومية": "entity",
    "نوع الفعالية": "event_type",
    "عنوان الفعالية": "title",
    "تاريخ بداية الفعالية": "start_date",
    "تاريخ نهاية الفعالية": "end_date",
    "نوع الموقع": "venue_type",
    "موقع الفعالية": "venue",
    "المدينة": "city",
    "رقم الطلب": "request_id",
}


def load_and_profile():
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig", dtype=str)
    df = df.rename(columns=COLS)

    # quick look at the data before checking it
    print(f"rows: {len(df)}, columns: {len(df.columns)}")
    print("nulls per column:")
    print(df.isnull().sum().to_string())
    print("duplicate rows:", df.duplicated().sum())
    print("duplicate request ids:", df["request_id"].duplicated().sum())

    # remove exact duplicates before the checks
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        logger.info(f"dropped {before - len(df)} duplicate row(s)")
    return df


def quality_checks(df):
    # Checks based on the 6 dimensions
    results = {}
    dates_start = pd.to_datetime(df["start_date"], format="%m/%d/%Y", errors="coerce")
    dates_end = pd.to_datetime(df["end_date"], format="%m/%d/%Y", errors="coerce")

    # 1 completeness: important columns should not be empty
    nulls = df[["entity", "title", "request_id"]].isnull().sum().sum()
    results["completeness"] = (nulls == 0, f"{nulls} nulls in important columns")

    # 2 accuracy: event duration should make sense (0 to 90 days)
    duration = (dates_end - dates_start).dt.days
    bad = ((duration < 0) | (duration > 90)).sum()
    results["accuracy"] = (bad == 0, f"{bad} events with weird duration")

    # 3 consistency: one request id should have one city only
    multi = (df.groupby("request_id")["city"].nunique() > 1).sum()
    results["consistency"] = (multi == 0, f"{multi} request ids with more than one city")

    # 4 timeliness: dates should be inside 2026-2027
    out = ((dates_start < "2026-01-01") | (dates_start > "2027-12-31")).sum()
    results["timeliness"] = (out == 0, f"{out} events outside 2026-2027")

    # 5 uniqueness: request id should not repeat
    dup = df["request_id"].duplicated().sum()
    results["uniqueness"] = (dup == 0, f"{dup} duplicated request ids")

    # 6 validity: request id format GOV-YYYY-NNNN
    bad_ids = (~df["request_id"].str.match(r"^GOV-\d{4}-\d{4}$", na=False)).sum()
    results["validity"] = (bad_ids == 0, f"{bad_ids} request ids with wrong format")

    print("\nquality report:")
    for name, (ok, detail) in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {detail}")
        if ok:
            logger.success(f"{name} passed")
        else:
            logger.warning(f"{name} failed: {detail}")

    passed = all(ok for ok, d in results.values())
    return passed


def run_great_expectations(df):
    # same checks but with the real great expectations library
    import great_expectations as gx
    import great_expectations.expectations as gxe

    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("my_source")
    asset = source.add_dataframe_asset(name="events")
    batch_def = asset.add_batch_definition_whole_dataframe("all")

    suite = context.suites.add(gx.ExpectationSuite(name="events_suite"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="request_id"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="entity"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="city"))
    suite.add_expectation(gxe.ExpectColumnValuesToMatchRegex(
        column="request_id", regex=r"^GOV-\d{4}-\d{4}$"))

    validation = context.validation_definitions.add(
        gx.ValidationDefinition(name="events_validation", data=batch_def, suite=suite))
    checkpoint = context.checkpoints.add(
        gx.Checkpoint(name="events_checkpoint", validation_definitions=[validation]))

    result = checkpoint.run(batch_parameters={"dataframe": df})
    print(f"\ngreat expectations result: success={result.success}")
    for run in result.run_results.values():
        for r in run["results"]:
            status = "PASS" if r["success"] else "FAIL"
            print(f"  [{status}] {r['expectation_config']['type']}")
    return result.success


def emit_lineage(passed, row_count):
    # openlineage events so we can track the pipeline runs
    from openlineage.client import OpenLineageClient
    from openlineage.client.transport.file import FileConfig, FileTransport
    from openlineage.client.event_v2 import RunEvent, RunState, Run, Job
    from openlineage.client.uuid import generate_new_uuid

    os.makedirs("lineage", exist_ok=True)
    transport = FileTransport(FileConfig(log_file_path="lineage/events.log"))
    client = OpenLineageClient(transport=transport)

    job = Job(namespace="capstone", name="quality_gate")
    run = Run(runId=str(generate_new_uuid()))
    now = datetime.now(UTC).isoformat()

    client.emit(RunEvent(eventType=RunState.START, eventTime=now,
                         run=run, job=job, producer="capstone-project"))
    end_state = RunState.COMPLETE if passed else RunState.FAIL
    client.emit(RunEvent(eventType=end_state, eventTime=now,
                         run=run, job=job, producer="capstone-project"))
    logger.success(f"openlineage events saved to lineage/events.log ({row_count} rows)")


def main():
    print("---- quality gate ----")

    df = load_and_profile()
    passed = quality_checks(df)
    gx_passed = run_great_expectations(df)

    # if the checks pass the data goes to production, if not to quarantine
    if passed and gx_passed:
        os.makedirs("production", exist_ok=True)
        df.to_parquet("production/events_clean.parquet", index=False)
        logger.success(f"quality gate PASSED, {len(df)} rows saved to production/")
    else:
        os.makedirs("quarantine", exist_ok=True)
        df.to_csv("quarantine/events_failed.csv", index=False, encoding="utf-8-sig")
        logger.error("quality gate FAILED, data went to quarantine/")

    emit_lineage(passed and gx_passed, len(df))
    return passed and gx_passed


if __name__ == "__main__":
    main()
