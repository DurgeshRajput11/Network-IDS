from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone

from kafka import KafkaProducer


def generate_features(size: int, attack_probability: float) -> tuple[list[float], int]:
    attack = 1 if random.random() < attack_probability else 0
    if attack:
        # Higher values to mimic suspicious traffic profile
        features = [round(random.uniform(0.6, 1.2), 4) for _ in range(size)]
    else:
        # Lower values to mimic benign traffic profile
        features = [round(random.uniform(0.0, 0.6), 4) for _ in range(size)]
    return features, attack


def main() -> None:
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC", "network-traffic")
    feature_size = int(os.getenv("FEATURE_SIZE", "20"))
    interval_seconds = float(os.getenv("PRODUCER_INTERVAL_SECONDS", "1.0"))
    attack_probability = float(os.getenv("ATTACK_PROBABILITY", "0.25"))

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"[producer] sending to topic={topic}")
    try:
        while True:
            features, intended_label = generate_features(feature_size, attack_probability)
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "simulator",
                "features": features,
                "simulated_label": intended_label,
            }
            producer.send(topic, value=event).get(timeout=10)
            print(json.dumps(event))
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("[producer] stopped")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
