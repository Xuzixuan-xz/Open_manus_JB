"""JobPilotFlow — deterministic multi-agent orchestrator for job application assistance."""
import json
import time
from typing import Any, Dict, Optional

from app.agent.jobpilot.cover_letter import CoverLetterAgent
from app.agent.jobpilot.interview import InterviewAgent
from app.agent.jobpilot.jd_parser import JDParserAgent
from app.agent.jobpilot.resume_optimizer import ResumeOptimizerAgent
from app.agent.jobpilot.review import ReviewAgent
from app.flow.base import BaseFlow
from app.logger import logger


class JobPilotFlow(BaseFlow):
    """Sequential multi-agent flow for job application assistance.

    Orchestrates five specialized agents in order:
      1. JDParserAgent       — parse JD → structured analysis
      2. ResumeOptimizerAgent — match resume to JD → optimization report
      3. InterviewAgent      — generate interview prep kit
      4. CoverLetterAgent    — draft cover letter & email
      5. ReviewAgent         — audit all materials → readiness score

    The output of each step is injected into the next agent's prompt via a shared
    ``context`` dict, keeping agents stateless and decoupled.

    Input format (JSON string or plain text):
        {
            "jd_text": "...",          # raw JD text (required unless jd_url given)
            "jd_url":  "...",          # URL of the job posting (optional)
            "resume_text": "...",      # resume as plain text (required unless resume_path given)
            "resume_path": "...",      # path to a PDF/DOCX resume file (optional)
            "company_name": "...",     # company name (optional)
            "company_url":  "..."      # company website URL (optional)
        }
    """

    class Config:
        arbitrary_types_allowed = True

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls) -> "JobPilotFlow":
        """Create a JobPilotFlow with a placeholder primary agent."""
        # BaseFlow requires at least one agent in __init__.
        # The actual per-step agents are instantiated fresh inside execute().
        placeholder = JDParserAgent()
        return cls(agents={"jd_parser": placeholder})

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def execute(self, input_text: str) -> str:  # noqa: C901
        """Run the full JobPilot pipeline.

        Args:
            input_text: JSON string or free-form text describing the job and candidate.
                        If JSON, fields ``jd_text``/``jd_url`` and ``resume_text``/``resume_path``
                        are used; otherwise the whole string is treated as the JD text.

        Returns:
            A comprehensive Markdown report consolidating all agent outputs.
        """
        context = self._parse_input(input_text)
        results: Dict[str, str] = {}

        # ── Step 1: Parse JD ──────────────────────────────────────────
        logger.info("🔍 [JobPilot] Step 1/5 — JD Parser")
        jd_prompt = self._build_jd_prompt(context)
        try:
            jd_result = await self._run_fresh_agent(JDParserAgent, jd_prompt, "JDParser")
        except Exception as e:
            jd_result = f"[JDParser failed: {e}]"
            logger.error(f"[JobPilot/JDParser] unhandled error: {e}")
        results["jd_analysis"] = jd_result
        context["jd_analysis"] = jd_result

        # ── Step 2: Resume Optimizer ──────────────────────────────────
        logger.info("📄 [JobPilot] Step 2/5 — Resume Optimizer")
        resume_prompt = self._build_resume_prompt(context)
        try:
            resume_result = await self._run_fresh_agent(
                ResumeOptimizerAgent, resume_prompt, "ResumeOptimizer"
            )
        except Exception as e:
            resume_result = f"[ResumeOptimizer failed: {e}]"
            logger.error(f"[JobPilot/ResumeOptimizer] unhandled error: {e}")
        results["resume_report"] = resume_result
        context["resume_report"] = resume_result

        # ── Step 3: Interview Prep ────────────────────────────────────
        logger.info("🎤 [JobPilot] Step 3/5 — Interview Prep")
        interview_prompt = self._build_interview_prompt(context)
        try:
            interview_result = await self._run_fresh_agent(
                InterviewAgent, interview_prompt, "InterviewPrep"
            )
        except Exception as e:
            interview_result = f"[InterviewPrep failed: {e}]"
            logger.error(f"[JobPilot/InterviewPrep] unhandled error: {e}")
        results["interview_kit"] = interview_result
        context["interview_kit"] = interview_result

        # ── Step 4: Cover Letter ──────────────────────────────────────
        logger.info("✉️  [JobPilot] Step 4/5 — Cover Letter")
        cover_prompt = self._build_cover_letter_prompt(context)
        try:
            cover_result = await self._run_fresh_agent(
                CoverLetterAgent, cover_prompt, "CoverLetter"
            )
        except Exception as e:
            cover_result = f"[CoverLetter failed: {e}]"
            logger.error(f"[JobPilot/CoverLetter] unhandled error: {e}")
        results["application_docs"] = cover_result
        context["application_docs"] = cover_result

        # ── Step 5: Review ────────────────────────────────────────────
        logger.info("🔎 [JobPilot] Step 5/5 — Review")
        review_prompt = self._build_review_prompt(context)
        try:
            review_result = await self._run_fresh_agent(
                ReviewAgent, review_prompt, "Review"
            )
        except Exception as e:
            review_result = f"[Review failed: {e}]"
            logger.error(f"[JobPilot/Review] unhandled error: {e}")
        results["final_review"] = review_result

        # ── Assemble final report ─────────────────────────────────────
        report = self._build_final_report(context, results)
        logger.info("✅ [JobPilot] Pipeline complete.")
        return report

    # ------------------------------------------------------------------
    # Agent runner
    # ------------------------------------------------------------------

    @staticmethod
    async def _run_fresh_agent(
        agent_cls: Any, prompt: str, step_name: str
    ) -> str:
        """Instantiate a fresh agent, run it with the prompt, return its output.

        The agent's actual analysis is stored as assistant message content in its
        memory.  ``agent.run()`` only returns tool-call traces (e.g. "Step 1:
        Observed output of cmd …"), so we extract the last non-empty assistant
        message instead to get the real LLM-generated content.
        """
        agent = agent_cls()
        try:
            await agent.run(prompt)

            # Extract the last assistant message that contains real content.
            # assistant messages may carry both `content` (the analysis text)
            # and `tool_calls` (e.g. terminate); we want the content part.
            content = ""
            for msg in reversed(agent.memory.messages):
                if msg.role == "assistant" and msg.content and msg.content.strip():
                    content = msg.content.strip()
                    break

            if not content:
                content = f"[{step_name}: no output produced]"

            logger.info(f"[JobPilot/{step_name}] completed ({len(content)} chars)")
            return content
        except Exception as e:
            logger.error(f"[JobPilot/{step_name}] failed: {e}")
            return f"[{step_name} failed: {e}]"

    # ------------------------------------------------------------------
    # Input parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_input(input_text: str) -> Dict[str, str]:
        """Parse the input text into a context dict."""
        ctx: Dict[str, str] = {}
        try:
            data = json.loads(input_text)
            if isinstance(data, dict):
                ctx = {k: str(v) for k, v in data.items()}
                return ctx
        except (json.JSONDecodeError, ValueError):
            pass

        # Plain text — treat the whole thing as the JD
        ctx["jd_text"] = input_text.strip()
        return ctx

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_jd_prompt(ctx: Dict[str, str]) -> str:
        parts = ["Please parse the following job description:"]
        if ctx.get("jd_url"):
            parts.append(
                f"\nThe JD may be available at this URL — try the web_scraper tool to fetch it:\n{ctx['jd_url']}"
                "\n(If web_scraper fails, use the JD text provided below instead.)"
            )
        if ctx.get("jd_text"):
            parts.append(f"\n--- JD TEXT ---\n{ctx['jd_text']}\n--- END JD ---")
        if not ctx.get("jd_url") and not ctx.get("jd_text"):
            parts.append("\n(No JD content provided — return an empty JSON result.)")
        if ctx.get("company_name"):
            parts.append(f"\nCompany: {ctx['company_name']}")
        if ctx.get("company_url"):
            parts.append(
                f"\nFor additional company context, you may scrape: {ctx['company_url']}"
            )
        parts.append(
            "\nOutput a JSON object with keys: required_skills, nice_to_have, "
            "responsibilities, culture_keywords, seniority."
        )
        return "\n".join(parts)

    @staticmethod
    def _build_resume_prompt(ctx: Dict[str, str]) -> str:
        parts = [
            "Please analyze the candidate's resume against the JD analysis below "
            "and produce an optimization report.",
            f"\n=== JD ANALYSIS ===\n{ctx.get('jd_analysis', 'N/A')}\n=== END JD ANALYSIS ===",
        ]
        if ctx.get("resume_path"):
            parts.append(
                f"\nThe resume file is at: {ctx['resume_path']}\n"
                "Use the doc_parser tool to read it first."
            )
        if ctx.get("resume_text"):
            parts.append(f"\n--- RESUME ---\n{ctx['resume_text']}\n--- END RESUME ---")
        if not ctx.get("resume_path") and not ctx.get("resume_text"):
            parts.append(
                "\n(No resume provided — provide a general analysis based on the JD alone.)"
            )
        return "\n".join(parts)

    @staticmethod
    def _build_interview_prompt(ctx: Dict[str, str]) -> str:
        resume_section = ""
        if ctx.get("resume_text"):
            resume_section = (
                f"\n=== RESUME ===\n{ctx['resume_text']}\n=== END RESUME ==="
            )
        return (
            "Generate a comprehensive interview preparation kit for the following role.\n"
            f"\n=== JD ANALYSIS ===\n{ctx.get('jd_analysis', 'N/A')}\n=== END JD ANALYSIS ==="
            f"{resume_section}"
            f"\n\nCompany: {ctx.get('company_name', 'Unknown')}"
        )

    @staticmethod
    def _build_cover_letter_prompt(ctx: Dict[str, str]) -> str:
        parts = [
            "Generate complete application materials (self-introductions, email, and cover letter).",
            f"\n=== JD ANALYSIS ===\n{ctx.get('jd_analysis', 'N/A')}\n=== END JD ANALYSIS ===",
        ]
        if ctx.get("resume_text"):
            parts.append(
                f"\n=== RESUME ===\n{ctx['resume_text']}\n=== END RESUME ==="
            )
        parts.append(
            f"\nCompany Name: {ctx.get('company_name', 'the target company')}"
        )
        if ctx.get("company_url"):
            parts.append(f"Company URL: {ctx['company_url']}")
        parts.append(
            "\nAfter generating the content, use the md_exporter tool to save it "
            "with filename 'cover_letter_and_email'."
        )
        return "\n".join(parts)

    @staticmethod
    def _build_review_prompt(ctx: Dict[str, str]) -> str:
        return (
            "Please review all of the following job application materials for authenticity, "
            "JD-alignment, and consistency. Provide your final assessment.\n\n"
            f"=== JD ANALYSIS ===\n{ctx.get('jd_analysis', 'N/A')}\n"
            f"=== RESUME OPTIMIZATION REPORT ===\n{ctx.get('resume_report', 'N/A')}\n"
            f"=== INTERVIEW KIT ===\n{ctx.get('interview_kit', 'N/A')}\n"
            f"=== APPLICATION DOCUMENTS ===\n{ctx.get('application_docs', 'N/A')}\n"
        )

    # ------------------------------------------------------------------
    # Report assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _build_final_report(
        ctx: Dict[str, str], results: Dict[str, str]
    ) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        company = ctx.get("company_name", "Target Company")
        jd_snippet = (ctx.get("jd_text") or ctx.get("jd_url") or "N/A")[:200]
        if len(jd_snippet) == 200:
            jd_snippet += "…"

        sections = [
            f"# JobPilot Application Report",
            f"\n**Generated:** {timestamp}  ",
            f"**Company:** {company}  ",
            f"**JD Preview:** {jd_snippet}",
            "\n---\n",
            "## 1. JD Analysis\n",
            results.get("jd_analysis", "_No output_"),
            "\n---\n",
            "## 2. Resume Optimization Report\n",
            results.get("resume_report", "_No output_"),
            "\n---\n",
            "## 3. Interview Preparation Kit\n",
            results.get("interview_kit", "_No output_"),
            "\n---\n",
            "## 4. Application Documents\n",
            results.get("application_docs", "_No output_"),
            "\n---\n",
            "## 5. Final Review & Readiness Score\n",
            results.get("final_review", "_No output_"),
            "\n---\n",
            "_Report generated by JobPilot — powered by OpenManus_",
        ]
        return "\n".join(sections)
