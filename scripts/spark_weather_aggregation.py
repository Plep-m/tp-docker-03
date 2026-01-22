from datetime import datetime, timezone
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, avg, sum as spark_sum, to_timestamp
from pyspark.sql.types import StructType, StructField, DoubleType, BooleanType, StringType
from hdfs import InsecureClient
import csv
from io import StringIO

KAFKA_BROKER = "kafka:9092"
KAFKA_TOPIC = "weather_transformed"
SPARK_MASTER = "spark://spark-master:7077"
CHECKPOINT_DIR = "/opt/spark/work-dir/checkpoints/weather_agg"

HDFS_WEB_URL = "http://namenode:9870"
HDFS_DIR = "/user/jovyan/weather_agg"
HDFS_USER = "root"


def ensure_hdfs_dir(hdfs_web_url, hdfs_path, user):
    """Create HDFS directory if it doesn't exist"""
    client = InsecureClient(hdfs_web_url, user=user)
    try:
        client.makedirs(hdfs_path)
    except Exception:
        pass  # Directory already exists


def write_csv_to_hdfs(hdfs_web_url, hdfs_path, user, header, rows):
    """Write CSV data to HDFS"""
    client = InsecureClient(hdfs_web_url, user=user)
    
    # Create CSV content in memory
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(rows)
    
    # Write to HDFS
    with client.write(hdfs_path, encoding='utf-8', overwrite=True) as hdfs_file:
        hdfs_file.write(output.getvalue())


def main():
    spark = SparkSession.builder \
        .appName("WeatherAggregation") \
        .master(SPARK_MASTER) \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    schema = StructType([
        StructField("temperature", DoubleType(), True),
        StructField("windspeed", DoubleType(), True),
        StructField("temp_f", DoubleType(), True),
        StructField("high_wind_alert", BooleanType(), True),
        StructField("time", StringType(), True)
    ])

    # Streaming read from Kafka
    raw_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "earliest") \
        .load()

    json_df = raw_df.selectExpr("CAST(value AS STRING) as json")
    parsed = json_df.select(from_json(col("json"), schema).alias("data")).select("data.*")
    parsed = parsed.withColumn("event_time", to_timestamp(col("time"), "yyyy-MM-dd'T'HH:mm"))

    agg = parsed.groupBy(
        window(col("event_time"), "1 minute")
    ).agg(
        avg("temperature").alias("avg_temp_c"),
        spark_sum(col("high_wind_alert").cast("int")).alias("alert_count")
    )

    ensure_hdfs_dir(HDFS_WEB_URL, HDFS_DIR, user=HDFS_USER)

    def write_batch_to_hdfs(batch_df, batch_id: int):
        if batch_df.rdd.isEmpty():
            return

        flat = batch_df.select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("avg_temp_c"),
            col("alert_count"),
        )

        rows = [
            (
                str(r["window_start"]),
                str(r["window_end"]),
                r["avg_temp_c"],
                r["alert_count"],
            )
            for r in flat.collect()
        ]

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        hdfs_path = f"{HDFS_DIR}/weather_agg_{ts}_batch{batch_id}.csv"

        write_csv_to_hdfs(
            hdfs_web_url=HDFS_WEB_URL,
            hdfs_path=hdfs_path,
            user=HDFS_USER,
            header=("window_start", "window_end", "avg_temp_c", "alert_count"),
            rows=rows,
        )

        print(f"[HDFS] Wrote CSV: {hdfs_path}", flush=True)

    query = (
        agg.writeStream
        .outputMode("update")
        .foreachBatch(write_batch_to_hdfs)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
