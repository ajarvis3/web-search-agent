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

from agent import run_agent

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Simple liveness check."""
    return jsonify({"status": "ok"}), 200


@app.route("/search", methods=["POST"])
def search():
    """Accept a JSON body with a *query* field and return the agent answer."""
    data = request.get_json(silent=True)
    if not data or "query" not in data:
        return jsonify({"error": "Request body must be JSON with a 'query' field."}), 400

    query = data["query"]
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "'query' must be a non-empty string."}), 400

    query = query.strip()
    try:
        answer = run_agent(query)
        return jsonify({"answer": answer}), 200
    except Exception as exc:
        logger.exception("Agent error for query %r", query)
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
