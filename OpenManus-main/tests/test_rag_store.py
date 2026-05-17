"""Unit tests for the JobPilot RAG store and retriever tool.

These tests run fully offline — no LLM calls, no ChromaDB required.
They use the forced in-memory backend via ``force_memory=True``.

Run from OpenManus-main/:
    PYTHONPATH=. python -m pytest -q tests/test_rag_store.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock optional heavy dependencies before any app import
# ---------------------------------------------------------------------------
for _mod in (
    "daytona",
    "browsergym",
    "browsergym.core",
    "browsergym.core.env",
    "browser_use",
    "browser_use.browser",
    "browser_use.browser.context",
    "browser_use.agent",
    "browser_use.agent.service",
    "browser_use.agent.views",
    "browser_use.controller",
    "browser_use.controller.service",
    "browser_use.dom",
    "browser_use.dom.service",
    "playwright",
    "playwright.async_api",
    "crawl4ai",
    "crawl4ai.async_webcrawler",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Ensure a valid config/config.toml exists
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.toml"

_TEST_CONFIG_CONTENT = """\
[llm]
model = "gpt-4o-mini"
base_url = "https://api.openai.com/v1/"
api_key = "test-key"
max_tokens = 4096
temperature = 0.0
api_type = "openai"
api_version = ""

[llm.vision]
model = "gpt-4o-mini"
base_url = "https://api.openai.com/v1/"
api_key = "test-key"
max_tokens = 4096
temperature = 0.0
api_type = "openai"
api_version = ""

[mcp]
server_reference = "app.mcp.server"

[runflow]
use_data_analysis_agent = false

[jobpilot]
enable = true
output_dir = "workspace/jobpilot"
max_interview_questions = 10
language = "zh"

[daytona]
daytona_api_key = "test-daytona-key"
"""


def _ensure_config() -> None:
    if not _CONFIG_PATH.exists():
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(_TEST_CONFIG_CONTENT)


_ensure_config()

# Now safe to import app modules
from app.tool.jobpilot.rag_store import (  # noqa: E402
    RAGStore,
    _BUILTIN_SEEDS,
    get_rag_store,
    reset_rag_store,
)
from app.tool.jobpilot.rag_retriever import (  # noqa: E402
    CATEGORY_COVER_LETTER,
    CATEGORY_INTERVIEW_QA,
    CATEGORY_RESUME_BULLETS,
    CATEGORY_SKILL_GRAPH,
    RAGRetrieverTool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before each test."""
    reset_rag_store()
    yield
    reset_rag_store()


@pytest.fixture
def mem_store() -> RAGStore:
    """Return a fresh in-memory RAGStore."""
    return RAGStore(force_memory=True)


# ---------------------------------------------------------------------------
# RAGStore — basic add / retrieve
# ---------------------------------------------------------------------------


class TestRAGStoreInMemory:
    def test_retrieve_returns_empty_for_unknown_category(self, mem_store: RAGStore):
        results = mem_store.retrieve("anything", "nonexistent_category")
        assert results == []

    def test_add_and_retrieve(self, mem_store: RAGStore):
        docs = [
            "Built FastAPI REST service handling 50K req/day.",
            "Implemented Redis caching reducing DB load by 60%.",
            "Deployed ML model with TorchServe achieving 40ms p99 latency.",
        ]
        mem_store.add_documents(docs, CATEGORY_RESUME_BULLETS)
        results = mem_store.retrieve("FastAPI REST service API", CATEGORY_RESUME_BULLETS, top_k=1)
        assert len(results) == 1
        assert "FastAPI" in results[0]

    def test_retrieve_respects_top_k(self, mem_store: RAGStore):
        docs = [f"Document number {i}" for i in range(20)]
        mem_store.add_documents(docs, CATEGORY_INTERVIEW_QA)
        results = mem_store.retrieve("document", CATEGORY_INTERVIEW_QA, top_k=3)
        assert len(results) <= 3

    def test_retrieve_orders_by_relevance(self, mem_store: RAGStore):
        docs = [
            "Unrelated content about cooking recipes.",
            "Designed distributed cache system with Redis for high-throughput backend service.",
            "Basic introduction to gardening.",
        ]
        mem_store.add_documents(docs, CATEGORY_RESUME_BULLETS)
        results = mem_store.retrieve(
            "Redis cache backend distributed system", CATEGORY_RESUME_BULLETS, top_k=2
        )
        assert "Redis" in results[0]

    def test_auto_seeding_on_first_retrieve(self, mem_store: RAGStore):
        """Auto-seeding should populate the store when a category is first queried."""
        # Cold store — no documents yet
        assert mem_store._backend.count(CATEGORY_RESUME_BULLETS) == 0
        results = mem_store.retrieve("Python backend", CATEGORY_RESUME_BULLETS)
        # After retrieve, category should be seeded
        assert mem_store._backend.count(CATEGORY_RESUME_BULLETS) > 0
        assert len(results) > 0

    def test_no_double_seeding(self, mem_store: RAGStore):
        """A second retrieve on the same category should not add more built-in seeds."""
        mem_store.retrieve("backend", CATEGORY_RESUME_BULLETS)
        count_after_first = mem_store._backend.count(CATEGORY_RESUME_BULLETS)
        mem_store.retrieve("frontend", CATEGORY_RESUME_BULLETS)
        count_after_second = mem_store._backend.count(CATEGORY_RESUME_BULLETS)
        assert count_after_first == count_after_second

    def test_clear_resets_category(self, mem_store: RAGStore):
        mem_store.add_documents(["doc1"], CATEGORY_SKILL_GRAPH)
        mem_store.clear(CATEGORY_SKILL_GRAPH)
        assert mem_store._backend.count(CATEGORY_SKILL_GRAPH) == 0

    def test_seed_from_list_replaces_in_memory(self, mem_store: RAGStore):
        mem_store.add_documents(["old doc"], CATEGORY_COVER_LETTER)
        new_docs = ["fresh doc A", "fresh doc B"]
        mem_store.seed_from_list(new_docs, CATEGORY_COVER_LETTER)
        results = mem_store.retrieve("fresh", CATEGORY_COVER_LETTER, top_k=5)
        assert any("fresh" in r for r in results)

    def test_empty_query_still_returns(self, mem_store: RAGStore):
        mem_store.add_documents(["some document"], CATEGORY_INTERVIEW_QA)
        # Empty query tokens — all docs have equal score 0, returns up to top_k
        results = mem_store.retrieve("", CATEGORY_INTERVIEW_QA, top_k=5)
        assert isinstance(results, list)

    def test_add_empty_list_is_noop(self, mem_store: RAGStore):
        mem_store.add_documents([], CATEGORY_RESUME_BULLETS)
        assert mem_store._backend.count(CATEGORY_RESUME_BULLETS) == 0


# ---------------------------------------------------------------------------
# RAGStore — built-in seeds
# ---------------------------------------------------------------------------


class TestBuiltinSeeds:
    def test_all_four_categories_have_seeds(self):
        for cat in (
            CATEGORY_RESUME_BULLETS,
            CATEGORY_INTERVIEW_QA,
            CATEGORY_SKILL_GRAPH,
            CATEGORY_COVER_LETTER,
        ):
            assert cat in _BUILTIN_SEEDS
            assert len(_BUILTIN_SEEDS[cat]) > 0, f"No seeds for category '{cat}'"

    def test_resume_bullets_contain_quantifiable_results(self):
        # Good bullet points should contain numbers
        bullets = _BUILTIN_SEEDS[CATEGORY_RESUME_BULLETS]
        has_number = any(any(c.isdigit() for c in b) for b in bullets)
        assert has_number

    def test_interview_qa_contain_star_framework(self):
        qa_items = _BUILTIN_SEEDS[CATEGORY_INTERVIEW_QA]
        has_star = any("STAR" in item or "Situation" in item for item in qa_items)
        assert has_star

    def test_skill_graph_covers_multiple_roles(self):
        graph_items = _BUILTIN_SEEDS[CATEGORY_SKILL_GRAPH]
        text = " ".join(graph_items).lower()
        assert "backend" in text
        assert "ml" in text or "machine learning" in text

    def test_cover_letter_templates_have_hook_and_closing(self):
        templates = _BUILTIN_SEEDS[CATEGORY_COVER_LETTER]
        text = " ".join(templates).lower()
        assert "hook" in text or "opening" in text
        assert "closing" in text or "closing" in text


# ---------------------------------------------------------------------------
# RAGStore — singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_rag_store_returns_same_instance(self):
        store_a = get_rag_store(force_memory=True)
        store_b = get_rag_store()  # should return the already-created instance
        assert store_a is store_b

    def test_reset_rag_store_clears_singleton(self):
        store_a = get_rag_store(force_memory=True)
        reset_rag_store()
        store_b = get_rag_store(force_memory=True)
        assert store_a is not store_b


# ---------------------------------------------------------------------------
# RAGRetrieverTool
# ---------------------------------------------------------------------------


class TestRAGRetrieverTool:
    @pytest.fixture(autouse=True)
    def _use_memory_backend(self):
        """Ensure singleton uses the in-memory backend for all tool tests."""
        reset_rag_store()
        get_rag_store(force_memory=True)
        yield
        reset_rag_store()

    @pytest.mark.asyncio
    async def test_valid_retrieval_returns_success(self):
        tool = RAGRetrieverTool()
        result = await tool.execute(
            query="Python backend API design",
            category=CATEGORY_RESUME_BULLETS,
            top_k=3,
        )
        assert not result.error
        assert result.output
        assert "Retrieved" in result.output

    @pytest.mark.asyncio
    async def test_invalid_category_returns_error(self):
        tool = RAGRetrieverTool()
        result = await tool.execute(query="anything", category="bad_category")
        assert result.error

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self):
        tool = RAGRetrieverTool()
        result = await tool.execute(query="   ", category=CATEGORY_INTERVIEW_QA)
        assert result.error

    @pytest.mark.asyncio
    async def test_top_k_clamped_to_10(self):
        tool = RAGRetrieverTool()
        result = await tool.execute(
            query="skill", category=CATEGORY_SKILL_GRAPH, top_k=999
        )
        assert not result.error

    @pytest.mark.asyncio
    async def test_all_valid_categories_work(self):
        tool = RAGRetrieverTool()
        for cat in (
            CATEGORY_RESUME_BULLETS,
            CATEGORY_INTERVIEW_QA,
            CATEGORY_SKILL_GRAPH,
            CATEGORY_COVER_LETTER,
        ):
            result = await tool.execute(query="engineer skills", category=cat, top_k=2)
            assert not result.error, f"Category '{cat}' returned error: {result.error}"

    @pytest.mark.asyncio
    async def test_tool_metadata(self):
        tool = RAGRetrieverTool()
        assert tool.name == "rag_retriever"
        assert "category" in tool.parameters["properties"]
        assert "query" in tool.parameters["properties"]
        assert "top_k" in tool.parameters["properties"]

    @pytest.mark.asyncio
    async def test_output_format_includes_index(self):
        tool = RAGRetrieverTool()
        result = await tool.execute(
            query="FastAPI backend service Python",
            category=CATEGORY_RESUME_BULLETS,
            top_k=2,
        )
        assert not result.error
        assert "[1]" in result.output

    @pytest.mark.asyncio
    async def test_no_results_returns_success_with_message(self):
        """An empty category should return success, not error."""
        tool = RAGRetrieverTool()
        # Seed a fresh store with an empty category (no docs added)
        reset_rag_store()
        store = get_rag_store(force_memory=True)
        store._seeded_categories.add(CATEGORY_RESUME_BULLETS)  # suppress auto-seed

        result = await tool.execute(
            query="anything", category=CATEGORY_RESUME_BULLETS, top_k=3
        )
        assert not result.error
        assert "No documents found" in result.output


# ---------------------------------------------------------------------------
# Agent smoke-tests: check that RAGRetrieverTool is wired into each agent
# ---------------------------------------------------------------------------


class TestAgentRAGWiring:
    """Verify that all four agents have RAGRetrieverTool in their tool collections."""

    @pytest.fixture(autouse=True)
    def mock_llm(self):
        """Patch LLM.__init__ to prevent tiktoken downloads."""
        from unittest.mock import patch

        with patch("app.llm.LLM.__init__", return_value=None):
            yield

    def _tool_names(self, agent) -> set:
        return {t.name for t in agent.available_tools.tools}

    def test_jd_parser_has_rag_retriever(self):
        from app.agent.jobpilot.jd_parser import JDParserAgent

        agent = JDParserAgent()
        assert "rag_retriever" in self._tool_names(agent)

    def test_resume_optimizer_has_rag_retriever(self):
        from app.agent.jobpilot.resume_optimizer import ResumeOptimizerAgent

        agent = ResumeOptimizerAgent()
        assert "rag_retriever" in self._tool_names(agent)

    def test_interview_agent_has_rag_retriever(self):
        from app.agent.jobpilot.interview import InterviewAgent

        agent = InterviewAgent()
        assert "rag_retriever" in self._tool_names(agent)

    def test_cover_letter_agent_has_rag_retriever(self):
        from app.agent.jobpilot.cover_letter import CoverLetterAgent

        agent = CoverLetterAgent()
        assert "rag_retriever" in self._tool_names(agent)
