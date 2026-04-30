from __future__ import annotations

# Exposes celery app for worker startup:
# uv run celery -A src.mlops.celery_worker worker --loglevel=info
from .tasks import celery_app

