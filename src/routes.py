"""HTTP routes for querying the web-search agent."""

import logging

from flask import Blueprint, jsonify, request

from src.agent import run_agent

logger = logging.getLogger(__name__)

routes = Blueprint("routes", __name__)

@routes.route("/health", methods=["GET"])
def health():
    """Simple liveness check."""
    return jsonify({"status": "ok"}), 200

@routes.route("/search", methods=["POST"])
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

