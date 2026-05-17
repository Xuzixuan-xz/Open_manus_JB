"""Integration tests for the JobPilot multi-agent system.

These tests are designed to run fully offline (no LLM API calls).
The LLM is mocked at the module level so that all agent.run() calls
exercise prompt-building and flow orchestration without network I/O.

Run from OpenManus-main/:
    PYTHONPATH=. python -m pytest -q tests/test_jobpilot_integration.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock heavy/optional third-party modules that are not needed for these tests
# and may not be installed in all environments (e.g. CI).
# This must happen BEFORE any app module is imported.
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
# Ensure a valid config/config.toml exists before any app module is imported.
# This mirrors the pattern used across the OpenManus test suite.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.toml"
_CONFIG_EXAMPLE = _PROJECT_ROOT / "config" / "config.example.toml"


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
from app.agent.jobpilot import (  # noqa: E402
    CoverLetterAgent,
    InterviewAgent,
    JDParserAgent,
    ResumeOptimizerAgent,
    ReviewAgent,
)
from app.flow.jobpilot_flow import JobPilotFlow  # noqa: E402
from app.tool.jobpilot.doc_parser import DocParserTool  # noqa: E402
from app.tool.jobpilot.md_exporter import MarkdownExporterTool  # noqa: E402
from app.tool.jobpilot.web_scraper import WebScraperTool  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_JD = """
Software Engineer Intern — AI Platform
Company: TechCorp
Requirements:
- Python (required)
- Machine Learning (required)
- SQL (required)
- PyTorch preferred
Responsibilities:
- Develop ML pipeline components
- Collaborate with senior engineers
Seniority: Intern
Culture: fast-paced, collaborative, data-driven
"""

SAMPLE_RESUME = """
Jane Doe | jane@example.com | github.com/janedoe

Education: BSc Computer Science, Tsinghua University (2022–2026)

Skills: Python, TensorFlow, SQL, Git, Linux

Projects:
- NLP Sentiment Classifier: Built a BERT-based text classifier achieving 92% accuracy on IMDB dataset.
- Recommendation System: Implemented collaborative filtering with 10K user dataset.
"""

SAMPLE_JD_ANALYSIS = json.dumps({
    "required_skills": ["Python", "Machine Learning", "SQL"],
    "nice_to_have": ["PyTorch"],
    "responsibilities": ["Develop ML pipeline components", "Collaborate with senior engineers"],
    "culture_keywords": ["fast-paced", "collaborative", "data-driven"],
    "seniority": "intern",
})

SAMPLE_CONTEXT = {
    "jd_text": SAMPLE_JD,
    "resume_text": SAMPLE_RESUME,
    "company_name": "TechCorp",
}


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Provide a temporary workspace directory."""
    ws = tmp_path / "workspace" / "jobpilot"
    ws.mkdir(parents=True)
    return ws


# ---------------------------------------------------------------------------
# Helper: mock agent.run() to return a canned response
# ---------------------------------------------------------------------------

def _make_mock_run(return_value: str = "Mock agent output"):
    return AsyncMock(return_value=return_value)


# ===========================================================================
# 1. Agent prompt-building tests (no LLM needed)
# ===========================================================================

class TestFlowPromptBuilders:
    """Unit tests for JobPilotFlow prompt construction helpers."""

    def test_parse_input_json(self):
        payload = json.dumps(SAMPLE_CONTEXT)
        ctx = JobPilotFlow._parse_input(payload)
        assert ctx["jd_text"] == SAMPLE_JD
        assert ctx["resume_text"] == SAMPLE_RESUME
        assert ctx["company_name"] == "TechCorp"

    def test_parse_input_plain_text(self):
        plain = "We are hiring a Python engineer with 2 years of experience."
        ctx = JobPilotFlow._parse_input(plain)
        assert ctx["jd_text"] == plain

    def test_build_jd_prompt_with_text(self):
        ctx = {"jd_text": SAMPLE_JD, "company_name": "TechCorp"}
        prompt = JobPilotFlow._build_jd_prompt(ctx)
        assert "JD TEXT" in prompt
        assert SAMPLE_JD in prompt
        assert "TechCorp" in prompt

    def test_build_jd_prompt_with_url(self):
        ctx = {"jd_url": "https://example.com/jobs/1"}
        prompt = JobPilotFlow._build_jd_prompt(ctx)
        assert "web_scraper" in prompt
        assert "https://example.com/jobs/1" in prompt

    def test_build_resume_prompt(self):
        ctx = {
            "jd_analysis": SAMPLE_JD_ANALYSIS,
            "resume_text": SAMPLE_RESUME,
        }
        prompt = JobPilotFlow._build_resume_prompt(ctx)
        assert "JD ANALYSIS" in prompt
        assert "RESUME" in prompt
        assert "Python" in prompt  # from JD analysis

    def test_build_resume_prompt_no_resume(self):
        ctx = {"jd_analysis": SAMPLE_JD_ANALYSIS}
        prompt = JobPilotFlow._build_resume_prompt(ctx)
        assert "No resume provided" in prompt

    def test_build_resume_prompt_with_path(self):
        ctx = {"jd_analysis": SAMPLE_JD_ANALYSIS, "resume_path": "/tmp/resume.pdf"}
        prompt = JobPilotFlow._build_resume_prompt(ctx)
        assert "doc_parser" in prompt
        assert "/tmp/resume.pdf" in prompt

    def test_build_interview_prompt(self):
        ctx = {"jd_analysis": SAMPLE_JD_ANALYSIS, "resume_text": SAMPLE_RESUME, "company_name": "TechCorp"}
        prompt = JobPilotFlow._build_interview_prompt(ctx)
        assert "JD ANALYSIS" in prompt
        assert "RESUME" in prompt

    def test_build_cover_letter_prompt(self):
        ctx = {
            "jd_analysis": SAMPLE_JD_ANALYSIS,
            "resume_text": SAMPLE_RESUME,
            "company_name": "TechCorp",
        }
        prompt = JobPilotFlow._build_cover_letter_prompt(ctx)
        assert "TechCorp" in prompt
        assert "md_exporter" in prompt

    def test_build_review_prompt(self):
        ctx = {
            "jd_analysis": SAMPLE_JD_ANALYSIS,
            "resume_report": "Match score: 80",
            "interview_kit": "Q: Tell me about yourself",
            "application_docs": "Dear TechCorp...",
        }
        prompt = JobPilotFlow._build_review_prompt(ctx)
        assert "JD ANALYSIS" in prompt
        assert "RESUME OPTIMIZATION REPORT" in prompt
        assert "INTERVIEW KIT" in prompt
        assert "APPLICATION DOCUMENTS" in prompt

    def test_build_final_report_structure(self):
        results = {
            "jd_analysis": "JD analysis here",
            "resume_report": "Resume report here",
            "interview_kit": "Interview kit here",
            "application_docs": "Docs here",
            "final_review": "Review here",
        }
        ctx = {"jd_text": SAMPLE_JD[:50], "company_name": "TechCorp"}
        report = JobPilotFlow._build_final_report(ctx, results)

        assert "# JobPilot Application Report" in report
        assert "## 1. JD Analysis" in report
        assert "## 2. Resume Optimization Report" in report
        assert "## 3. Interview Preparation Kit" in report
        assert "## 4. Application Documents" in report
        assert "## 5. Final Review" in report
        assert "TechCorp" in report
        assert "OpenManus" in report


# ===========================================================================
# 2. Tool unit tests
# ===========================================================================

class TestDocParserTool:
    """Unit tests for DocParserTool."""

    @pytest.mark.asyncio
    async def test_parse_raw_text(self):
        tool = DocParserTool()
        result = await tool.execute(text="Hello, world!")
        assert result.error is None
        assert "Hello, world!" in result.output

    @pytest.mark.asyncio
    async def test_parse_txt_file(self, tmp_path):
        f = tmp_path / "resume.txt"
        f.write_text("Name: Jane Doe\nSkills: Python, SQL")
        tool = DocParserTool()
        result = await tool.execute(file_path=str(f))
        assert result.error is None
        assert "Jane Doe" in result.output

    @pytest.mark.asyncio
    async def test_missing_file(self):
        tool = DocParserTool()
        result = await tool.execute(file_path="/nonexistent/file.pdf")
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_args_returns_error(self):
        tool = DocParserTool()
        result = await tool.execute()
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_text_takes_precedence_over_path(self, tmp_path):
        f = tmp_path / "resume.txt"
        f.write_text("file content")
        tool = DocParserTool()
        result = await tool.execute(file_path=str(f), text="inline content")
        assert "inline content" in result.output


class TestMarkdownExporterTool:
    """Unit tests for MarkdownExporterTool."""

    @pytest.mark.asyncio
    async def test_export_creates_file(self, tmp_path):
        with patch("app.tool.jobpilot.md_exporter.config") as mock_cfg:
            mock_cfg.workspace_root = tmp_path
            tool = MarkdownExporterTool()
            result = await tool.execute(content="# Test Report\n\nHello!", filename="test_report")

        assert result.error is None
        assert "test_report.md" in result.output

        saved_files = list(tmp_path.rglob("*.md"))
        assert len(saved_files) == 1
        assert "# Test Report" in saved_files[0].read_text()

    @pytest.mark.asyncio
    async def test_empty_content_fails(self, tmp_path):
        tool = MarkdownExporterTool()
        result = await tool.execute(content="")
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_default_filename_has_timestamp(self, tmp_path):
        with patch("app.tool.jobpilot.md_exporter.config") as mock_cfg:
            mock_cfg.workspace_root = tmp_path
            tool = MarkdownExporterTool()
            result = await tool.execute(content="Some content")

        assert result.error is None
        saved_files = list(tmp_path.rglob("*.md"))
        assert any("jobpilot_report_" in f.name for f in saved_files)

    @pytest.mark.asyncio
    async def test_filename_sanitization(self, tmp_path):
        with patch("app.tool.jobpilot.md_exporter.config") as mock_cfg:
            mock_cfg.workspace_root = tmp_path
            tool = MarkdownExporterTool()
            result = await tool.execute(content="Content", filename="My Report 2024!")

        assert result.error is None
        saved_files = list(tmp_path.rglob("*.md"))
        assert all(
            c.isalnum() or c in ("-", "_", ".")
            for f in saved_files
            for c in f.stem
        )


class TestWebScraperTool:
    """Unit tests for WebScraperTool."""

    def test_invalid_url_schema(self):
        """No network call needed — URL validation is synchronous logic."""
        import asyncio
        tool = WebScraperTool()
        result = asyncio.run(tool.execute(url="ftp://not-http.com"))
        assert result.error is not None
        assert "http" in result.error.lower()

    @pytest.mark.asyncio
    async def test_mocked_http_fetch(self):
        tool = WebScraperTool()
        html = (
            "<html><body><main><h1>Software Engineer</h1>"
            "<p>We need Python skills.</p></main></body></html>"
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        with patch("app.tool.jobpilot.web_scraper.requests.get", return_value=mock_response):
            result = await tool.execute(url="https://example.com/jobs/1")

        assert result.error is None
        assert "Software Engineer" in result.output
        assert "Python" in result.output


# ===========================================================================
# 3. Agent instantiation and tool wiring tests
# ===========================================================================

# Fixture that patches LLM creation to avoid network calls (tiktoken downloads)
@pytest.fixture(autouse=False)
def mock_llm():
    """Patch LLM.__init__ to prevent tiktoken downloads during tests."""
    with patch("app.llm.LLM.__init__", return_value=None):
        yield


class TestAgentInstantiation:
    """Verify agents can be instantiated and have the expected tools."""

    @pytest.fixture(autouse=True)
    def _mock_llm(self, mock_llm):
        pass

    def test_jd_parser_has_web_tools(self):
        agent = JDParserAgent()
        assert "web_scraper" in agent.available_tools.tool_map
        assert "web_search" in agent.available_tools.tool_map
        assert "terminate" in agent.available_tools.tool_map

    def test_resume_optimizer_has_doc_parser(self):
        agent = ResumeOptimizerAgent()
        assert "doc_parser" in agent.available_tools.tool_map
        assert "terminate" in agent.available_tools.tool_map

    def test_interview_agent_has_web_search(self):
        agent = InterviewAgent()
        assert "web_search" in agent.available_tools.tool_map
        assert "terminate" in agent.available_tools.tool_map

    def test_cover_letter_agent_has_md_exporter(self):
        agent = CoverLetterAgent()
        assert "md_exporter" in agent.available_tools.tool_map
        assert "terminate" in agent.available_tools.tool_map

    def test_review_agent_has_terminate_only(self):
        agent = ReviewAgent()
        tool_names = set(agent.available_tools.tool_map.keys())
        assert "terminate" in tool_names
        # ReviewAgent is intentionally lean — no external tool calls needed
        assert "web_scraper" not in tool_names

    def test_agent_names_are_unique(self):
        agents = [
            JDParserAgent(),
            ResumeOptimizerAgent(),
            InterviewAgent(),
            CoverLetterAgent(),
            ReviewAgent(),
        ]
        names = [a.name for a in agents]
        assert len(names) == len(set(names)), "Agent names must be unique"

    def test_agent_descriptions_non_empty(self):
        agents = [
            JDParserAgent(),
            ResumeOptimizerAgent(),
            InterviewAgent(),
            CoverLetterAgent(),
            ReviewAgent(),
        ]
        for agent in agents:
            assert agent.description, f"{agent.name} must have a description"


# ===========================================================================
# 4. JobPilotFlow orchestration tests (agents mocked)
# ===========================================================================

class TestJobPilotFlow:
    """Integration tests for JobPilotFlow orchestration with mocked agents."""

    @pytest.fixture(autouse=True)
    def _mock_llm(self, mock_llm):
        pass

    @pytest.mark.asyncio
    async def test_flow_create(self):
        flow = JobPilotFlow.create()
        assert flow is not None
        assert isinstance(flow, JobPilotFlow)

    @pytest.mark.asyncio
    async def test_flow_executes_all_five_steps(self):
        """Verify all 5 agent steps are called in order."""
        call_order = []

        async def mock_run(cls, prompt, step_name):
            call_order.append(step_name)
            return f"Mock output from {step_name}"

        flow = JobPilotFlow.create()
        with patch.object(JobPilotFlow, "_run_fresh_agent", side_effect=mock_run):
            report = await flow.execute(json.dumps(SAMPLE_CONTEXT))

        assert call_order == [
            "JDParser",
            "ResumeOptimizer",
            "InterviewPrep",
            "CoverLetter",
            "Review",
        ], f"Unexpected step order: {call_order}"

        # Report should contain all 5 sections
        assert "## 1. JD Analysis" in report
        assert "## 2. Resume Optimization Report" in report
        assert "## 3. Interview Preparation Kit" in report
        assert "## 4. Application Documents" in report
        assert "## 5. Final Review" in report

    @pytest.mark.asyncio
    async def test_flow_context_propagation(self):
        """Verify that each step's output is available to later steps."""
        captured_prompts: dict[str, str] = {}

        async def mock_run(cls, prompt, step_name):
            captured_prompts[step_name] = prompt
            return f"OUTPUT_OF_{step_name}"

        flow = JobPilotFlow.create()
        with patch.object(JobPilotFlow, "_run_fresh_agent", side_effect=mock_run):
            await flow.execute(json.dumps(SAMPLE_CONTEXT))

        # ResumeOptimizer prompt must contain JD analysis from JDParser
        assert "OUTPUT_OF_JDParser" in captured_prompts["ResumeOptimizer"]
        # InterviewPrep prompt must contain JD analysis
        assert "OUTPUT_OF_JDParser" in captured_prompts["InterviewPrep"]
        # Review prompt must contain all previous outputs
        review_prompt = captured_prompts["Review"]
        assert "OUTPUT_OF_JDParser" in review_prompt
        assert "OUTPUT_OF_ResumeOptimizer" in review_prompt
        assert "OUTPUT_OF_InterviewPrep" in review_prompt
        assert "OUTPUT_OF_CoverLetter" in review_prompt

    @pytest.mark.asyncio
    async def test_flow_handles_agent_failure_gracefully(self):
        """A failing step should not crash the whole pipeline."""
        call_count = [0]

        async def flaky_run(cls, prompt, step_name):
            call_count[0] += 1
            if step_name == "JDParser":
                raise RuntimeError("Simulated LLM failure")
            return f"Output of {step_name}"

        flow = JobPilotFlow.create()
        with patch.object(JobPilotFlow, "_run_fresh_agent", side_effect=flaky_run):
            report = await flow.execute(json.dumps(SAMPLE_CONTEXT))

        # All 5 steps attempted
        assert call_count[0] == 5
        # Failure message in the report
        assert "JDParser failed" in report

    @pytest.mark.asyncio
    async def test_flow_plain_text_input(self):
        """Flow should accept plain JD text (not JSON)."""
        async def mock_run(cls, prompt, step_name):
            return "ok"

        flow = JobPilotFlow.create()
        with patch.object(JobPilotFlow, "_run_fresh_agent", side_effect=mock_run):
            report = await flow.execute("We are hiring a Python developer.")

        assert "JobPilot Application Report" in report

    @pytest.mark.asyncio
    async def test_flow_with_jd_url_builds_correct_prompt(self):
        """When jd_url is given, JD prompt should instruct agent to use web_scraper."""
        captured: dict[str, str] = {}

        async def mock_run(cls, prompt, step_name):
            captured[step_name] = prompt
            return "ok"

        flow = JobPilotFlow.create()
        ctx = {"jd_url": "https://example.com/jobs/42"}
        with patch.object(JobPilotFlow, "_run_fresh_agent", side_effect=mock_run):
            await flow.execute(json.dumps(ctx))

        assert "web_scraper" in captured["JDParser"]
        assert "https://example.com/jobs/42" in captured["JDParser"]

    @pytest.mark.asyncio
    async def test_flow_with_resume_path_builds_correct_prompt(self):
        """When resume_path is given, resume prompt should instruct agent to use doc_parser."""
        captured: dict[str, str] = {}

        async def mock_run(cls, prompt, step_name):
            captured[step_name] = prompt
            return "ok"

        flow = JobPilotFlow.create()
        ctx = {"jd_text": SAMPLE_JD, "resume_path": "/tmp/resume.pdf"}
        with patch.object(JobPilotFlow, "_run_fresh_agent", side_effect=mock_run):
            await flow.execute(json.dumps(ctx))

        assert "doc_parser" in captured["ResumeOptimizer"]
        assert "/tmp/resume.pdf" in captured["ResumeOptimizer"]
