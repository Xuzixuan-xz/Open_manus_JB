"""JobPilotFlow — deterministic multi-agent orchestrator for job application assistance."""
import json
import re
import time
from typing import Any, Dict, Optional

from app.agent.jobpilot.cover_letter import CoverLetterAgent
from app.agent.jobpilot.interview import InterviewAgent
from app.agent.jobpilot.jd_parser import JDParserAgent
from app.agent.jobpilot.resume_optimizer import ResumeOptimizerAgent
from app.agent.jobpilot.review import ReviewAgent
from app.flow.base import BaseFlow
from app.logger import logger


# Unicode ranges used for CJK language detection
_CJK_UNIFIED_START = "\u4e00"
_CJK_UNIFIED_END = "\u9fff"
_CJK_EXTENSION_A_START = "\u3400"
_CJK_EXTENSION_A_END = "\u4dbf"
_CJK_RATIO_THRESHOLD = 0.05  # minimum fraction of CJK chars to classify text as Chinese

# Matches a bare terminate directive that the LLM sometimes writes as plain text
# (e.g. 'terminate with status "success"') instead of using the terminate tool.
# Such lines must never appear in the user-visible report.
_TERMINATE_LINE_RE = re.compile(
    r"^\s*terminate\s+with\s+status\s+[\"']?(success|failure)[\"']?\s*$",
    re.IGNORECASE,
)


def _strip_terminate_lines(text: str) -> str:
    """Remove stray 'terminate with status …' lines from *text*.

    The LLM occasionally writes the terminate directive as plain text in its
    response content instead of (or in addition to) issuing a proper tool call.
    Such lines must never appear in the user-visible report.

    Args:
        text: Raw content string that may contain bare terminate directives.

    Returns:
        The cleaned text with all matching lines removed and leading/trailing
        whitespace stripped.  Returns an empty string if *text* is empty or
        consists solely of terminate directive lines.
    """
    cleaned = "\n".join(
        line for line in text.splitlines() if not _TERMINATE_LINE_RE.match(line)
    )
    return cleaned.strip()


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

        If no assistant text content is found (the agent wrote everything into
        an ``md_exporter`` tool call instead), we fall back to extracting the
        ``content`` argument from the most recent ``md_exporter`` invocation.
        """
        agent = agent_cls()
        try:
            await agent.run(prompt)

            # 1) Extract the last assistant message that contains real content.
            # assistant messages may carry both `content` (the analysis text)
            # and `tool_calls` (e.g. terminate); we want the content part.
            content = ""
            for msg in reversed(agent.memory.messages):
                stripped = _strip_terminate_lines(msg.content or "")
                if msg.role == "assistant" and stripped:
                    content = stripped
                    break

            # 2) Fallback: if the agent put its output entirely inside an
            # md_exporter tool call (no response content), extract from there.
            if not content:
                for msg in reversed(agent.memory.messages):
                    if msg.role == "assistant" and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if tool_call.function.name == "md_exporter":
                                try:
                                    args = json.loads(
                                        tool_call.function.arguments or "{}"
                                    )
                                    extracted = _strip_terminate_lines(
                                        args.get("content", "")
                                    )
                                    if extracted:
                                        content = extracted
                                except (json.JSONDecodeError, ValueError):
                                    pass
                            if content:
                                break
                    if content:
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
    def _detect_language(text: str) -> str:
        """Return 'zh' if *text* contains a significant proportion of CJK characters, else 'en'."""
        if not text:
            return "en"
        cjk_count = sum(
            1
            for ch in text
            if _CJK_UNIFIED_START <= ch <= _CJK_UNIFIED_END
            or _CJK_EXTENSION_A_START <= ch <= _CJK_EXTENSION_A_END
        )
        return "zh" if cjk_count / max(len(text), 1) > _CJK_RATIO_THRESHOLD else "en"

    @staticmethod
    def _lang_instruction(ctx: Dict[str, str]) -> str:
        """Return a language instruction line based on the detected JD language."""
        lang = ctx.get("lang", "en")
        if lang == "zh":
            return "\n\n**语言要求：请用中文（简体）输出你的全部分析和建议，所有标题、正文、示例均使用中文。**"
        return "\n\n**Language requirement: Respond entirely in English.**"

    @staticmethod
    def _parse_input(input_text: str) -> Dict[str, str]:
        """Parse the input text into a context dict."""
        ctx: Dict[str, str] = {}
        try:
            data = json.loads(input_text)
            if isinstance(data, dict):
                ctx = {k: str(v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            # Plain text — treat the whole thing as the JD
            ctx["jd_text"] = input_text.strip()

        # Detect language from JD text (or URL hint) and store in context
        jd_sample = ctx.get("jd_text") or ctx.get("jd_url") or ""
        ctx["lang"] = JobPilotFlow._detect_language(jd_sample)
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
            parts.append(
                "\n(No JD content provided — return a JSON result with empty lists for all fields "
                "and seniority set to \"unknown\".)"
            )
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
        parts.append(JobPilotFlow._lang_instruction(ctx))
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
        parts.append(JobPilotFlow._lang_instruction(ctx))
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
            f"{JobPilotFlow._lang_instruction(ctx)}"
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
        # For cover letters the self-intro is intentionally bilingual;
        # the framing/instructions should still follow the JD language.
        if ctx.get("lang") == "zh":
            parts.append(
                "\n\n**语言要求：除双语自我介绍部分外（中文版 + 英文版），其余所有内容（邮件、求职信、章节标题等）请用中文撰写。**"
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
            f"\n{JobPilotFlow._lang_instruction(ctx)}"
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

        lang = ctx.get("lang", "en")
        if lang == "zh":
            title = "# JobPilot 求职申请报告"
            label_generated = "**生成时间：**"
            label_company = "**公司：**"
            label_jd_preview = "**职位描述摘要：**"
            sec1 = "## 1. 职位描述解析"
            sec2 = "## 2. 简历优化报告"
            sec3 = "## 3. 面试准备工具包"
            sec4 = "## 4. 求职材料"
            sec5 = "## 5. 最终审核与准备度评分"
            footer = "_报告由 JobPilot 生成 — 基于 OpenManus 驱动_"
        else:
            title = "# JobPilot Application Report"
            label_generated = "**Generated:**"
            label_company = "**Company:**"
            label_jd_preview = "**JD Preview:**"
            sec1 = "## 1. JD Analysis"
            sec2 = "## 2. Resume Optimization Report"
            sec3 = "## 3. Interview Preparation Kit"
            sec4 = "## 4. Application Documents"
            sec5 = "## 5. Final Review & Readiness Score"
            footer = "_Report generated by JobPilot — powered by OpenManus_"

        sections = [
            title,
            f"\n{label_generated} {timestamp}  ",
            f"{label_company} {company}  ",
            f"{label_jd_preview} {jd_snippet}",
            "\n---\n",
            f"{sec1}\n",
            results.get("jd_analysis", "_No output_"),
            "\n---\n",
            f"{sec2}\n",
            results.get("resume_report", "_No output_"),
            "\n---\n",
            f"{sec3}\n",
            results.get("interview_kit", "_No output_"),
            "\n---\n",
            f"{sec4}\n",
            results.get("application_docs", "_No output_"),
            "\n---\n",
            f"{sec5}\n",
            results.get("final_review", "_No output_"),
            "\n---\n",
            footer,
        ]
        return "\n".join(sections)
