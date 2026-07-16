# kafka producer + consumer with schema validation

import json
import logging
import re
import os
import socket
import time
from datetime import datetime

# hide the noisy kafka retry logs when no broker is running
logging.getLogger("kafka").setLevel(logging.CRITICAL)

import pandas as pd
from loguru import logger
from pydantic import BaseModel, field_validator, ValidationError

CSV_FILE = "Public_Sector_Events_Q2_2026_CSV.csv"
TOPIC = "gov_events"

# rename the arabic columns to english names so its easier to use in code
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


def load_data():
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig", dtype=str)
    df = df.rename(columns=COLS)
    logger.info(f"loaded {len(df)} rows from the csv")
    return df


# the schema, every message must pass this before we accept it
class EventSchema(BaseModel):
    entity: str
    event_type: str
    title: str
    start_date: str
    end_date: str
    venue_type: str
    venue: str
    city: str
    request_id: str

    @field_validator("entity", "title", "city")
    @classmethod
    def check_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("empty field")
        return v.strip()

    @field_validator("request_id")
    @classmethod
    def check_request_id(cls, v):
        # should look like GOV-2026-0040
        if not re.match(r"^GOV-\d{4}-\d{4}$", v.strip()):
            raise ValueError("bad request_id: " + v)
        return v.strip()

    @field_validator("start_date", "end_date")
    @classmethod
    def check_date(cls, v):
        # dates in the file are like 1/4/2026
        datetime.strptime(v.strip(), "%m/%d/%Y")
        return v.strip()


class MockKafka:
    def __init__(self):
        self.messages = []

    def send(self, topic, value):
        self.messages.append(value)

    def flush(self):
        pass


def make_messages(df):
    msgs = df.to_dict(orient="records")
    # add 2 bad messages on purpose to test that the validation catches them
    bad1 = dict(msgs[0])
    bad1["request_id"] = "WRONG-ID"
    bad2 = dict(msgs[1])
    bad2["start_date"] = "not a date"
    return msgs + [bad1, bad2]


def validate_messages(messages):
    accepted = []
    rejected = 0
    for i, m in enumerate(messages):
        try:
            event = EventSchema(**m)
            accepted.append(event.model_dump())
            logger.debug(f"accepted message {i}")
        except ValidationError as e:
            rejected += 1
            logger.warning(f"rejected message {i}: {e.errors()[0]['msg']}")
    return accepted, rejected


def kafka_is_up():
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(("localhost", 9092))
        return True
    except OSError:
        return False
    finally:
        s.close()


def run_real_kafka(messages):
    # this only works if kafka is running on localhost:9092
    from kafka import KafkaProducer, KafkaConsumer

    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    for m in messages:
        producer.send(TOPIC, m)
    producer.flush()
    producer.close()
    time.sleep(1)

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers="localhost:9092",
        auto_offset_reset="earliest",
        group_id="my_group",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=10000,
    )
    received = [r.value for r in consumer]
    consumer.close()
    return received


def main():
    print("---- Deliverable 1: ingestion ----")
    df = load_data()
    messages = make_messages(df)
    logger.info(f"producing {len(messages)} messages to topic {TOPIC}")

    try:
        if not kafka_is_up():
            raise ConnectionError("no kafka broker on localhost:9092")
        received = run_real_kafka(messages)
        logger.info("used real kafka broker")
    except Exception as e:
        logger.warning(f"kafka not available ({e}), using mock instead")
        mock = MockKafka()
        for m in messages:
            mock.send(TOPIC, m)
        received = mock.messages

    accepted, rejected = validate_messages(received)

    # save the good rows, deliverable 2 will read this file
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(accepted).to_csv("data/validated_events.csv", index=False)
    logger.success(f"saved {len(accepted)} valid rows to data/validated_events.csv")

    print(f"total messages: {len(received)}")
    print(f"accepted: {len(accepted)}")
    print(f"rejected: {rejected}")


if __name__ == "__main__":
    main()
