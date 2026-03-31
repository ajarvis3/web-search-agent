"""
Web-search agent backed by Google Gemini and LangChain.

The agent is given a web-search tool (Tavily). The Gemini LLM decides
for each query whether its built-in knowledge is sufficient or whether
it needs to call the search tool to find up-to-date information.
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from typing import List
import threading
import logging
from tavily import TavilyClient

load_dotenv()

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions accurately. "
    "You have access to a web_search tool. "
    "First, consider whether you already know the answer from your training data. "
    "If the answer is well-established and not time-sensitive, answer directly "
    "without using the tool. "
    "If the answer may be outdated, requires current information, or you are "
    "uncertain, use the web_search tool to look it up before answering."
    "Clearly state in your response whether you used the web_search tool or not."
)

_CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "chroma"))
_CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "web_search_agent")
_CHROMA_INDEX_MARKER = _CHROMA_DIR / ".indexed_wiki_v1"

logger = logging.getLogger(__name__)


def _build_vector_store() -> Chroma:
    """Create a persistent Chroma store for future retrieval features."""
    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # Use a Hugging Face sentence-transformers model by default
    embedding_model = os.getenv(
        "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(
        collection_name=_CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(_CHROMA_DIR),
    )


def index_wikipedia_pages(page: str, chunk_size: int = 1000, chunk_overlap: int = 200, persist: bool = True) -> int:
    """Load the given Wikipedia page titles/ids, split into chunks and add to Chroma.

    This operation is potentially slow and network-bound (it will download pages
    and compute embeddings). It is NOT called automatically by agent initialization;
    call it explicitly when you want to populate the vector store.

    Returns the number of document chunks added.
    """
    if not page:
        raise ValueError("`pages` must be a non-empty string to query")

    # Load pages from Wikipedia
    docs = WikipediaLoader(query=page, load_max_docs=1).load()

    # Split into chunks for better retrieval granularity
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    # Add chunks to the vector store
    vs = get_vector_store()
    try:
        vs.add_documents(chunks)
    except Exception:
        # Some Chroma wrappers may expose different methods; try a fallback
        try:
            for d in chunks:
                vs.add_document(d)
        except Exception:
            raise

    # Persist to disk if supported
    if persist and hasattr(vs, "persist"):
        try:
            vs.persist()
        except Exception:
            # non-fatal: keep the in-memory store if persist fails
            pass

    return len(chunks)


def _chroma_is_indexed() -> bool:
    """Return True if a marker file indicates Wikipedia was indexed already."""
    try:
        return _CHROMA_INDEX_MARKER.exists()
    except Exception:
        return False


def _maybe_background_index() -> None:
    if _chroma_is_indexed():
        logger.info("Chroma wiki index marker present, skipping startup indexing.")
        return

    pages = ["George Washington", "Abraham Lincoln", "Machine learning"]
    if not pages:
        logger.info("No pages configured for INDEX_WIKI_PAGES; skipping indexing.")
        return

    def _worker():
        try:
            total = 0
            for page in pages:
                logger.info("Starting background Wikipedia indexing for %s", page)
                added = index_wikipedia_pages(page)
                total += added
            # write marker on success
            try:
                _CHROMA_INDEX_MARKER.write_text(f"indexed:{total}\n")
            except Exception:
                logger.exception("Failed to write chroma index marker file")
            logger.info("Background indexing finished: %d chunks added", total)
        except Exception:
            logger.exception("Background Wikipedia indexing failed")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


@tool
def tavily_search(query: str) -> str:
    """Search the web using Tavily"""
    tavily_key = os.getenv("TAVILY_KEY")

    if tavily_key is None:
        raise RuntimeError("TAVILY_KEY not set")

    tavily_client = TavilyClient(api_key=tavily_key)
    response = tavily_client.search(query,search_depth='fast',max_results=3)
    return response.text

def start_background_indexing() -> None:
    try:
        _maybe_background_index()
    except Exception:
        logger.exception("Failed to start background indexing")


def _build_agent() -> Any:
    """Construct and return a compiled LangChain agent graph."""
    # Initialize persistence on first agent build; retrieval wiring can be added incrementally.
    get_vector_store()

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)

    tools = [tavily_search]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
    )


_agent: Any = None
_vector_store: Any = None


def get_vector_store() -> Any:
    """Return the singleton Chroma store, creating it on first call."""
    global _vector_store
    if _vector_store is None:
        _vector_store = _build_vector_store()
    return _vector_store


def get_agent() -> Any:
    """Return the singleton agent graph, creating it on first call."""
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def run_agent(query: str) -> str:
    """Run the agent for *query* and return its final answer string."""
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    agent = get_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query.strip()}]}
    )
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return ""
