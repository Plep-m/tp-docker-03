from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from kafka import KafkaConsumer
from hdfs import InsecureClient
import json
import traceback

KAFKA_TOPIC = "weather_transformed"
KAFKA_BROKER = "kafka:9092"
HDFS_DIR = "/user/jovyan/alerts"
HDFS_CLIENT = InsecureClient("http://namenode:9870", user="root")

def read_once_and_upload():
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=5000
    )
    
    try:
        msg = next(consumer)
        print(f"✓ Message received from Kafka: {msg.value}")
    except StopIteration:
        print("⚠ No messages in Kafka topic.")
        consumer.close()
        return
    
    record = msg.value
    consumer.close()
    
    json_data = json.dumps(record, indent=2)
    
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"alerts_{ts}.json"
    hdfs_path = f"{HDFS_DIR}/{filename}"

    try:
        print(f"Creating directory and writing to: {hdfs_path}")
        
        HDFS_CLIENT.makedirs(HDFS_DIR)
        
        with HDFS_CLIENT.write(hdfs_path, overwrite=False, encoding="utf-8") as writer:
            writer.write(json_data)
        
        print(f"✓ Upload successful: {hdfs_path}")
    except Exception as e:
        print("❌ HDFS ERROR:")
        traceback.print_exc()
        raise

with DAG(
    dag_id="weather_alert_direct_to_hdfs_timestamped",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["weather", "kafka", "hdfs"],
    description="Consume one weather message from Kafka and save to HDFS"
) as dag:
    
    read_once = PythonOperator(
        task_id="consume_one_message",
        python_callable=read_once_and_upload,
    )

    read_once
