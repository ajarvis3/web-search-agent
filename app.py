"""
Flask application entry point for the web-search agent.

Endpoints
---------
POST /search
    Body (JSON): {"query": "<question>"}
    Response (JSON): {"answer": "<answer>"} or {"error": "<message>"}

GET /health
    Response (JSON): {"status": "ok"}
"""

import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from src.agent import run_agent
from src.agent import start_background_indexing
from src.routes import routes

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.register_blueprint(routes)


@app.before_serving
def _start_optional_indexing():
    """Trigger optional background Wikipedia indexing on first request.

    Controlled by the INDEX_WIKI_ON_STARTUP env var. Non-blocking.
    """
    try:
        start_background_indexing()
    except Exception:
        logger.exception("Failed to start background indexing from Flask app")

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
