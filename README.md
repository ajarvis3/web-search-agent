# web-search-agent

A Flask-based web search agent powered by **Google Gemini** and **LangChain**.

When a query arrives the Gemini LLM first considers whether it can answer from
its own knowledge.  Only if the answer may be outdated or uncertain does the
agent invoke the **DuckDuckGo** search tool to look up current information.

---

## Architecture

```
POST /search  ──►  Flask (app.py)  ──►  LangChain agent (agent.py)
                                             │
                                    ┌────────┴────────┐
                                    │                 │
                              Gemini LLM        DuckDuckGo
                              (answers          search tool
                            from knowledge)   (live web search)
```

---

## Quick start

### 1 — Prerequisites

- Python 3.12+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Configure environment

Copy the example file and fill in your API key:

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=<your key>
```

### 4 — Run the server

```bash
python app.py
```

The server starts on `http://0.0.0.0:5000` by default.

---

## API

### `GET /health`

Liveness check.

**Response**
```json
{"status": "ok"}
```

---

### `POST /search`

Ask the agent a question.

**Request body** (JSON)

| Field   | Type   | Required | Description        |
|---------|--------|----------|--------------------|
| `query` | string | yes      | The question to ask |

**Example**

```bash
curl -X POST http://localhost:5000/search \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
```

**Successful response** (`200`)

```json
{"answer": "The capital of France is Paris."}
```

**Error response** (`400` / `500`)

```json
{"error": "..."}
```

---

## Running tests

```bash
pytest
```

---

## Environment variables

| Variable       | Default           | Description                        |
|----------------|-------------------|------------------------------------|
| `GOOGLE_API_KEY` | *(required)*    | Google Gemini API key              |
| `GEMINI_MODEL` | `gemini-2.5-flash`| Gemini model identifier            |
| `HUGGINGFACE_EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | Hugging Face sentence-transformers model for embeddings |
| `CHROMA_PERSIST_DIR` | `chroma` | Local directory for Chroma files |
| `CHROMA_COLLECTION` | `web_search_agent` | Chroma collection name |
| `FLASK_HOST`   | `0.0.0.0`         | Host address for the Flask server  |
| `FLASK_PORT`   | `5000`            | Port for the Flask server          |
| `FLASK_DEBUG`  | `false`           | Enable Flask debug mode            |
