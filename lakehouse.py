# Deliverable 2
# delta lakehouse with bronze / silver / gold + MERGE + schema enforcement

import os
import shutil

from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

BRONZE = "data/delta/bronze"
SILVER = "data/delta/silver"
GOLD = "data/delta/gold"

SCHEMA = StructType([
    StructField("entity", StringType()),
    StructField("event_type", StringType()),
    StructField("title", StringType()),
    StructField("start_date", StringType()),
    StructField("end_date", StringType()),
    StructField("venue_type", StringType()),
    StructField("venue", StringType()),
    StructField("city", StringType()),
    StructField("request_id", StringType()),
])


def get_spark():
    builder = (
        SparkSession.builder.appName("lakehouse")
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # the data is small (600 rows) so a few partitions is enough,
        # the default is 200 and that makes spark very slow on windows
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.ui.enabled", "false")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    # set it again here because getOrCreate can return an old session
    # that ignores the builder configs
    spark.conf.set("spark.sql.shuffle.partitions", "4")
    return spark


def main():
    print("---- Deliverable 2: lakehouse ----")

    # start clean every run
    if os.path.exists("data/delta"):
        shutil.rmtree("data/delta")

    spark = get_spark()

    # ---------- bronze ----------
    # bronze = the raw data as it is, we take the validated file from deliverable 1
    if os.path.exists("data/validated_events.csv"):
        df = spark.read.csv("data/validated_events.csv", header=True)
    else:
        logger.warning("run ingestion.py first, using the raw csv for now")
        df = spark.read.option("header", True).csv("Public_Sector_Events_Q2_2026_CSV.csv")
        old_names = df.columns
        for old, new in zip(old_names, [f.name for f in SCHEMA.fields]):
            df = df.withColumnRenamed(old, new)

    df = df.select([f.name for f in SCHEMA.fields])
    df.write.format("delta").mode("overwrite").save(BRONZE)
    logger.success(f"bronze saved with {df.count()} rows")

    # ---------- silver ----------
    # silver = cleaned data: real dates, no duplicates, fix extra spaces in title
    bronze_df = spark.read.format("delta").load(BRONZE)
    silver_df = (
        bronze_df
        .withColumn("start_date", F.to_date("start_date", "M/d/yyyy"))
        .withColumn("end_date", F.to_date("end_date", "M/d/yyyy"))
        .withColumn("title", F.regexp_replace(F.trim("title"), r"\s+", " "))
        .withColumn("duration_days", F.datediff("end_date", "start_date") + 1)
        .dropDuplicates(["request_id"])
    )
    logger.info("writing silver table...")
    silver_df.write.format("delta").mode("overwrite").save(SILVER)
    logger.success(f"silver saved with {silver_df.count()} rows (duplicates removed)")
    silver_df.select("request_id", "entity", "city", "start_date", "duration_days").show(5)

    # ---------- merge ----------
    # merge = update a row if it exists, insert it if it doesnt (upsert)
    # example: one event changed its venue + one new event got approved
    some_id = silver_df.first()["request_id"]
    updates = spark.createDataFrame(
        [
            ("x", "x", "x", "1/1/2026", "1/1/2026", "فنادق",
             "فندق الفيصلية (تحديث)", "الرياض", some_id),
            ("هيئة سدايا", "ورشة عمل", "ورشة هندسة البيانات", "7/20/2026",
             "7/22/2026", "فنادق", "كراون بلازا", "الرياض", "GOV-2026-8888"),
        ],
        SCHEMA,
    )
    updates = (updates
               .withColumn("start_date", F.to_date("start_date", "M/d/yyyy"))
               .withColumn("end_date", F.to_date("end_date", "M/d/yyyy"))
               .withColumn("duration_days", F.datediff("end_date", "start_date") + 1))

    logger.info("running the merge...")
    silver_table = DeltaTable.forPath(spark, SILVER)
    (silver_table.alias("t")
        .merge(updates.alias("u"), "t.request_id = u.request_id")
        .whenMatchedUpdate(set={"venue": "u.venue", "city": "u.city"})
        .whenNotMatchedInsertAll()
        .execute())
    logger.success(f"merge done: updated venue of {some_id} and inserted GOV-2026-8888")
    spark.read.format("delta").load(SILVER) \
        .filter(F.col("request_id").isin(some_id, "GOV-2026-8888")) \
        .select("request_id", "venue", "city").show(truncate=False)

    # ---------- gold ----------
    # gold = summary tables for reporting
    silver_df = spark.read.format("delta").load(SILVER)

    logger.info("building gold tables...")
    by_city = silver_df.groupBy("city").count().orderBy(F.desc("count"))
    by_city.write.format("delta").mode("overwrite").save(GOLD + "/events_by_city")
    print("events by city:")
    by_city.show(10)

    by_type = silver_df.groupBy("event_type").count().orderBy(F.desc("count"))
    by_type.write.format("delta").mode("overwrite").save(GOLD + "/events_by_type")
    print("events by type:")
    by_type.show(10)

    logger.success("gold tables saved")

    # ---------- schema enforcement (last step) ----------
    # try to write a row with an extra column, delta should reject it.
    # note: on windows this rejection can take a while, thats why its last
    print("testing schema enforcement (this write should fail)...")
    bad_schema = StructType(SCHEMA.fields + [StructField("budget", StringType())])
    bad_row = spark.createDataFrame(
        [("test", "test", "test", "1/1/2026", "1/1/2026", "test", "test", "الرياض",
          "GOV-2026-9999", "50000")],
        bad_schema)
    try:
        bad_row.write.format("delta").mode("append").save(BRONZE)
        logger.error("the write was accepted, schema enforcement did not work!")
    except Exception as e:
        logger.success("delta rejected the write because of the extra column, good")
        print("error was:", str(e).splitlines()[0])

    spark.stop()


if __name__ == "__main__":
    main()
