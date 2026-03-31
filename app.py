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

from dotenv import load_dotenv, dotenv_values
from pathlib import Path
from flask import Flask, jsonify, request

from src.agent import start_background_indexing
from src.routes import routes

# Load environment from .env, then allow local overrides from .env.local if present.
# .env.local is commonly used for machine-specific secrets and should override .env.
load_dotenv()
if os.path.exists('.env.local'):
    # override=True ensures values in .env.local replace ones from .env
    load_dotenv('.env.local', override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.register_blueprint(routes)


with app.app_context():
    try:
        if os.getenv("TAVILY_KEY", False):
            start_background_indexing()
    except Exception:
        logger.exception("Failed to start background indexing from Flask app")

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
