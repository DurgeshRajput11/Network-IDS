from __future__ import annotations

import json
import os
import uuid
import urllib.request


def http_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "kbn-xsrf": "true"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        content = resp.read().decode("utf-8")
        return json.loads(content) if content else {}


def create_data_view(kibana_url: str, pattern: str, time_field: str) -> str:
    response = http_json(
        "POST",
        f"{kibana_url}/api/data_views/data_view",
        {"data_view": {"title": pattern, "timeFieldName": time_field}},
    )
    return response["data_view"]["id"]


def create_visualization(
    kibana_url: str,
    title: str,
    vis_state: dict,
    search_source: dict,
) -> str:
    vis_id = str(uuid.uuid4())
    payload = {
        "attributes": {
            "title": title,
            "visState": json.dumps(vis_state),
            "uiStateJSON": "{}",
            "description": "",
            "version": 1,
            "kibanaSavedObjectMeta": {"searchSourceJSON": json.dumps(search_source)},
        }
    }
    http_json("POST", f"{kibana_url}/api/saved_objects/visualization/{vis_id}", payload)
    return vis_id


def create_dashboard(kibana_url: str, title: str, panel_ids: list[str]) -> None:
    panels = []
    for i, pid in enumerate(panel_ids):
        panels.append(
            {
                "panelIndex": str(i + 1),
                "gridData": {"x": (i % 2) * 24, "y": (i // 2) * 15, "w": 24, "h": 15, "i": str(i + 1)},
                "type": "visualization",
                "id": pid,
                "version": "8.14.3",
            }
        )

    dashboard_payload = {
        "attributes": {
            "title": title,
            "description": "Hybrid IDS monitoring dashboard",
            "panelsJSON": json.dumps(panels),
            "optionsJSON": json.dumps({"useMargins": True, "syncColors": False, "syncCursor": True, "syncTooltips": True}),
            "version": 1,
            "kibanaSavedObjectMeta": {"searchSourceJSON": json.dumps({"query": {"query": "", "language": "kuery"}, "filter": []})},
            "timeRestore": False,
        }
    }
    http_json("POST", f"{kibana_url}/api/saved_objects/dashboard", dashboard_payload)


def main() -> None:
    kibana_url = os.getenv("KIBANA_URL", "http://127.0.0.1:5601").rstrip("/")
    index_pattern = os.getenv("ELASTICSEARCH_INDEX_PATTERN", "ids-predictions-*")
    time_field = "@timestamp"

    data_view_id = create_data_view(kibana_url, index_pattern, time_field)
    search_source = {"index": data_view_id, "query": {"query": "", "language": "kuery"}, "filter": []}

    attack_count_vis = {
        "title": "Attack Count Over Time",
        "type": "histogram",
        "params": {"type": "histogram", "grid": {"categoryLines": False, "style": {"color": "#eee"}}, "addTooltip": True, "addLegend": True, "legendPosition": "right", "times": [], "addTimeMarker": False},
        "aggs": [
            {"id": "1", "enabled": True, "type": "count", "schema": "metric", "params": {}},
            {"id": "2", "enabled": True, "type": "date_histogram", "schema": "segment", "params": {"field": "@timestamp", "timeRange": {"from": "now-15m", "to": "now"}, "useNormalizedEsInterval": True, "interval": "auto", "drop_partials": False, "min_doc_count": 1, "extended_bounds": {}}},
        ],
    }
    attack_split_vis = {
        "title": "Attack vs Benign Split",
        "type": "pie",
        "params": {"type": "pie", "addTooltip": True, "addLegend": True, "legendPosition": "right", "isDonut": True, "labels": {"show": True, "values": True}},
        "aggs": [
            {"id": "1", "enabled": True, "type": "count", "schema": "metric", "params": {}},
            {"id": "2", "enabled": True, "type": "terms", "schema": "segment", "params": {"field": "attack", "size": 5, "order": "desc", "orderBy": "1"}},
        ],
    }
    confidence_vis = {
        "title": "Confidence Distribution",
        "type": "histogram",
        "params": {"type": "histogram", "addTooltip": True, "addLegend": False},
        "aggs": [
            {"id": "1", "enabled": True, "type": "count", "schema": "metric", "params": {}},
            {"id": "2", "enabled": True, "type": "histogram", "schema": "segment", "params": {"field": "confidence", "interval": 0.05, "min_doc_count": 1}},
        ],
    }
    sources_vis = {
        "title": "Top Suspicious Sources",
        "type": "horizontal_bar",
        "params": {"type": "horizontal_bar", "addTooltip": True, "addLegend": False},
        "aggs": [
            {"id": "1", "enabled": True, "type": "count", "schema": "metric", "params": {}},
            {"id": "2", "enabled": True, "type": "terms", "schema": "segment", "params": {"field": "source.keyword", "size": 10, "order": "desc", "orderBy": "1"}},
            {"id": "3", "enabled": True, "type": "filter", "schema": "group", "params": {"query": {"query_string": {"query": "attack:1"}}}},
        ],
    }

    panel_ids = [
        create_visualization(kibana_url, "Attack Count Over Time", attack_count_vis, search_source),
        create_visualization(kibana_url, "Attack vs Benign Split", attack_split_vis, search_source),
        create_visualization(kibana_url, "Confidence Distribution", confidence_vis, search_source),
        create_visualization(kibana_url, "Top Suspicious Sources", sources_vis, search_source),
    ]
    create_dashboard(kibana_url, "Hybrid IDS Monitoring Dashboard", panel_ids)
    print("Kibana data view and dashboard created successfully.")


if __name__ == "__main__":
    main()
