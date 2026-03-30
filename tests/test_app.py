"""
Unit tests for the Flask app and agent module.

The LangChain agent and the DuckDuckGo search tool are mocked so the tests
run without a live network connection or a real Gemini API key.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app():
    """Import app with a patched agent so no real LLM is invoked."""
    import app as flask_app
    flask_app.app.config["TESTING"] = True
    return flask_app.app.test_client()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200_and_ok(self):
        client = _make_app()
        response = client.get("/health")
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /search — input validation
# ---------------------------------------------------------------------------

class TestSearchValidation:
    def test_missing_body_returns_400(self):
        client = _make_app()
        response = client.post("/search", content_type="application/json", data="")
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_missing_query_field_returns_400(self):
        client = _make_app()
        response = client.post(
            "/search",
            content_type="application/json",
            data=json.dumps({"text": "hello"}),
        )
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_empty_query_returns_400(self):
        client = _make_app()
        response = client.post(
            "/search",
            content_type="application/json",
            data=json.dumps({"query": "   "}),
        )
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_non_string_query_returns_400(self):
        client = _make_app()
        response = client.post(
            "/search",
            content_type="application/json",
            data=json.dumps({"query": 42}),
        )
        assert response.status_code == 400
        assert "error" in response.get_json()


# ---------------------------------------------------------------------------
# /search — successful agent response
# ---------------------------------------------------------------------------

class TestSearchSuccess:
    def test_returns_answer_from_agent(self):
        client = _make_app()
        with patch("app.run_agent", return_value="Paris") as mock_agent:
            response = client.post(
                "/search",
                content_type="application/json",
                data=json.dumps({"query": "What is the capital of France?"}),
            )
        assert response.status_code == 200
        body = response.get_json()
        assert body == {"answer": "Paris"}
        mock_agent.assert_called_once_with("What is the capital of France?")

    def test_query_is_stripped_before_passing_to_agent(self):
        client = _make_app()
        with patch("app.run_agent", return_value="42") as mock_agent:
            client.post(
                "/search",
                content_type="application/json",
                data=json.dumps({"query": "  answer  "}),
            )
        # run_agent should receive the stripped version
        mock_agent.assert_called_once_with("answer")


# ---------------------------------------------------------------------------
# /search — agent error handling
# ---------------------------------------------------------------------------

class TestSearchError:
    def test_agent_exception_returns_500(self):
        client = _make_app()
        with patch("app.run_agent", side_effect=RuntimeError("LLM failure")):
            response = client.post(
                "/search",
                content_type="application/json",
                data=json.dumps({"query": "What happened?"}),
            )
        assert response.status_code == 500
        body = response.get_json()
        assert "error" in body
        assert "LLM failure" in body["error"]


# ---------------------------------------------------------------------------
# agent.run_agent — unit tests
# ---------------------------------------------------------------------------

class TestRunAgent:
    def test_run_agent_returns_output(self):
        """run_agent should return the last message's content from the agent result."""
        from src import agent as agent_module

        last_msg = MagicMock()
        last_msg.content = "42"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [last_msg]}

        with patch.object(agent_module, "get_agent", return_value=mock_agent):
            result = agent_module.run_agent("What is 6 times 7?")

        assert result == "42"
        mock_agent.invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "What is 6 times 7?"}]}
        )

    def test_run_agent_strips_query(self):
        from src import agent as agent_module

        last_msg = MagicMock()
        last_msg.content = "answer"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [last_msg]}

        with patch.object(agent_module, "get_agent", return_value=mock_agent):
            agent_module.run_agent("  hello  ")

        mock_agent.invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "hello"}]}
        )

    def test_run_agent_raises_on_empty_query(self):
        from src import agent as agent_module

        with pytest.raises(ValueError, match="empty"):
            agent_module.run_agent("")

    def test_run_agent_raises_on_whitespace_query(self):
        from src import agent as agent_module

        with pytest.raises(ValueError, match="empty"):
            agent_module.run_agent("   ")
