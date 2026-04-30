from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from kafka import KafkaConsumer, KafkaProducer


def post_prediction(api_url: str, features: list[float]) -> dict:
    payload = json.dumps({"features": features}).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    input_topic = os.getenv("KAFKA_TOPIC", "network-traffic")
    output_topic = os.getenv("KAFKA_ALERT_TOPIC", "ids-alerts")
    group_id = os.getenv("KAFKA_GROUP_ID", "ids-consumer-group")
    api_url = os.getenv("PREDICT_API_URL", "http://127.0.0.1:8000/predict")

    consumer = KafkaConsumer(
        input_topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="latest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"[consumer] listening on topic={input_topic}, api={api_url}")
    try:
        for msg in consumer:
            event = msg.value
            features = event.get("features", [])
            if not features:
                continue
            try:
                pred = post_prediction(api_url=api_url, features=features)
            except urllib.error.URLError as exc:
                print(f"[consumer] api error: {exc}")
                continue

            enriched = {"event": event, "prediction": pred}
            producer.send(output_topic, value=enriched).get(timeout=10)
            print(json.dumps(enriched))
    except KeyboardInterrupt:
        print("[consumer] stopped")
    finally:
        consumer.close()
        producer.close()


if __name__ == "__main__":
    main()
