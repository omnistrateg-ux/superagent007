"""
MOEX Agent v2 Web Dashboard

FastAPI-based web interface for monitoring signals and status.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .config import load_config
from .storage import connect, get_alerts

app = FastAPI(
    title="MOEX Agent v2",
    description="Trading Signal Generator for Moscow Exchange",
    version="2.0.0",
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Dashboard home page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MOEX Agent v2</title>
        <style>
            body { font-family: monospace; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .ok { background: #d4edda; }
            .warn { background: #fff3cd; }
            .error { background: #f8d7da; }
            a { color: #007bff; }
        </style>
    </head>
    <body>
        <h1>MOEX Agent v2</h1>
        <div class="status ok">Status: Running</div>
        <h2>API Endpoints</h2>
        <ul>
            <li><a href="/api/status">/api/status</a> - System status</li>
            <li><a href="/api/alerts">/api/alerts</a> - Recent alerts</li>
            <li><a href="/api/models">/api/models</a> - Model info</li>
            <li><a href="/docs">/docs</a> - API documentation</li>
        </ul>
    </body>
    </html>
    """


@app.get("/api/status")
async def get_status() -> Dict[str, Any]:
    """Get system status."""
    try:
        config = load_config()

        result = {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
            "config": {
                "tickers": len(config.tickers),
                "poll_seconds": config.poll_seconds,
                "p_threshold": config.p_threshold,
            },
        }

        # Database info
        if config.sqlite_path.exists():
            conn = connect(config.sqlite_path)
            cur = conn.execute("SELECT COUNT(*) as cnt FROM candles")
            result["database"] = {
                "candles": cur.fetchone()["cnt"],
            }
            cur = conn.execute("SELECT COUNT(*) as cnt FROM alerts")
            result["database"]["alerts"] = cur.fetchone()["cnt"]
            conn.close()
        else:
            result["database"] = {"status": "not initialized"}

        # Models info
        models_dir = Path("./models")
        meta_path = models_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            result["models"] = {
                "loaded": list(meta.keys()),
                "count": len(meta),
            }
        else:
            result["models"] = {"status": "not trained"}

        return result

    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/alerts")
async def api_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent alerts."""
    try:
        config = load_config()

        if not config.sqlite_path.exists():
            return []

        conn = connect(config.sqlite_path)
        alerts = get_alerts(conn, limit=limit)
        conn.close()

        return [
            {
                "id": a["id"],
                "created_ts": a["created_ts"],
                "secid": a["secid"],
                "direction": a["direction"],
                "horizon": a["horizon"],
                "p": a["p"],
                "signal_type": a["signal_type"],
                "entry": a["entry"],
                "take": a["take"],
                "stop": a["stop"],
                "anomaly_score": a["anomaly_score"],
                "sent": bool(a["sent"]),
            }
            for a in alerts
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def api_models() -> Dict[str, Any]:
    """Get model information."""
    models_dir = Path("./models")
    meta_path = models_dir / "meta.json"

    if not meta_path.exists():
        return {"status": "not trained", "models": {}}

    meta = json.loads(meta_path.read_text())

    return {
        "status": "ok",
        "models": {
            h: {
                "type": info.get("type"),
                "trained_at": info.get("trained_at"),
                "metrics": info.get("metrics", {}),
            }
            for h, info in meta.items()
        },
    }


@app.get("/api/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
