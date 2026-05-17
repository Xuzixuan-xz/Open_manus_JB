import re

from app.flow.base import BaseFlow


_SECTION_LABELS = {
    "jd": (
        "jd",
        "job description",
        "job requirements",
        "requirements",
        "岗位要求",
        "岗位描述",
        "职位描述",
        "任职要求",
    ),
    "background": (
        "background",
        "candidate background",
        "resume",
        "experience",
        "经历",
        "背景",
        "简历",
        "项目经历",
    ),
}
# Matches bullet markers and numbered list formats such as "- item", "* item",
# "1. item", "1)", "(1) item", and the corresponding full-width CJK variants.
_LIST_ITEM_RE = re.compile(r"^(?:[-*•]|(?:\d+[\.\)])|(?:[（(]?\d+[）)]))\s*")
_SECTION_LABEL_PATTERNS = {
    section: tuple(
        re.compile(rf"^{re.escape(label)}\s*:?\s*(.*)$", re.IGNORECASE)
        for label in labels
    )
    for section, labels in _SECTION_LABELS.items()
}


def _match_section_label(line: str) -> tuple[str | None, str]:
    normalized = line.strip().replace("：", ":")
    for section, patterns in _SECTION_LABEL_PATTERNS.items():
        for pattern in patterns:
            match = pattern.match(normalized)
            if match:
                return section, match.group(1).strip()
    return None, ""


def _clean_fact_item(value: str) -> str:
    return _LIST_ITEM_RE.sub("", value).strip(" \t-–—•")


def _normalize_fact_text(value: str) -> str:
    """Normalize text for case-insensitive, whitespace-tolerant fact matching."""
    return " ".join(value.split()).casefold()


def _split_inline_items(value: str) -> list[str]:
    if not value:
        return []
    segments = re.split(r"[;,；、]\s*", value)
    return [item for item in (_clean_fact_item(segment) for segment in segments) if item]


def _extract_structured_user_facts(input_text: str) -> dict[str, list[str]]:
    """Extract explicit JD/background facts from labeled user input.

    Rules:
    - A known section label starts a JD or background block.
    - Inline content on the same label line is captured immediately.
    - Bullet/numbered lines under the active section are captured as facts.
    - The first plain line after a section label is accepted as a fact to support
      single-line sections such as "JD: Java, Spring Boot".
    """
    facts = {"jd": [], "background": []}
    current_section: str | None = None
    accept_next_line_as_fact = False

    for raw_line in input_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            accept_next_line_as_fact = False
            continue

        section, inline_value = _match_section_label(stripped)
        if section:
            current_section = section
            accept_next_line_as_fact = True
            facts[section].extend(_split_inline_items(inline_value))
            continue

        if current_section is None:
            continue

        if _LIST_ITEM_RE.match(stripped):
            cleaned = _clean_fact_item(stripped)
            if cleaned:
                facts[current_section].append(cleaned)
            accept_next_line_as_fact = False
            continue

        if accept_next_line_as_fact:
            cleaned = _clean_fact_item(stripped)
            if cleaned:
                facts[current_section].append(cleaned)
            accept_next_line_as_fact = False
            continue

        current_section = None

    for section, values in facts.items():
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            # Preserve the user's original formatting while ignoring whitespace-only duplicates.
            key = " ".join(value.split())
            if key not in seen:
                seen.add(key)
                deduped.append(value)
        facts[section] = deduped

    return facts


def _format_fact_block(
    title: str, items: list[str], empty_text: str | None = None
) -> str:
    lines = [title]
    if items:
        lines.extend(f"- {item}" for item in items)
    elif empty_text is not None:
        lines.append(f"- {empty_text}")
    return "\n".join(lines)


def _build_grounding_context(
    input_text: str, facts: dict[str, list[str]] | None = None
) -> str:
    facts = facts or _extract_structured_user_facts(input_text)
    return "\n".join(
        [
            "[Grounding Context]",
            "User request:",
            input_text,
            "",
            "[Structured User Facts]",
            _format_fact_block(
                "[Explicit JD Facts]",
                facts["jd"],
                "No explicit JD-labeled facts detected.",
            ),
            "",
            _format_fact_block(
                "[Explicit Candidate Background]",
                facts["background"],
                "No explicit background-labeled facts detected.",
            ),
            "",
            "Treat every explicit fact above as user-provided ground truth. Do not mark those facts as unknown or ask the user to repeat them.",
        ]
    )


def _inject_coordinator_fact_guardrails(
    coordinator_plan: str, facts: dict[str, list[str]]
) -> str:
    """Append explicit user facts when coordinator output omits them."""
    explicit_jd = facts.get("jd", [])
    explicit_background = facts.get("background", [])
    if not explicit_jd and not explicit_background:
        return coordinator_plan

    normalized_plan = _normalize_fact_text(coordinator_plan)
    missing_facts = [
        fact
        for fact in (*explicit_jd, *explicit_background)
        if _normalize_fact_text(fact) not in normalized_plan
    ]
    if not missing_facts:
        return coordinator_plan

    guardrail_sections = [
        coordinator_plan.strip(),
        "",
        "[Coordinator Fact Guardrails]",
        "The following explicit user-provided facts are confirmed and must be preserved downstream.",
    ]
    if explicit_jd:
        guardrail_sections.append(_format_fact_block("[Explicit JD Facts]", explicit_jd))
    if explicit_background:
        if explicit_jd:
            guardrail_sections.append("")
        guardrail_sections.append(
            _format_fact_block("[Explicit Candidate Background]", explicit_background)
        )
    return "\n".join(guardrail_sections)


class JobPilotFlow(BaseFlow):
    """Deterministic multi-agent workflow for job/internship applications."""

    async def execute(self, input_text: str) -> str:
        facts = _extract_structured_user_facts(input_text)
        grounding_context = _build_grounding_context(input_text, facts=facts)
        coordinator = self._require_agent("coordinator")
        jd_agent = self._require_agent("jd_analysis")
        company_agent = self._require_agent("company_research")
        resume_agent = self._require_agent("resume_optimization")
        interview_agent = self._require_agent("interview")
        review_agent = self._require_agent("review")
        report_agent = self._require_agent("report")

        coordinator_plan = await coordinator.run(
            f"""
{grounding_context}

Create a concise, grounded job-application execution brief for downstream specialist agents.
Include confirmed facts, unknowns, and role-specific priorities.
"""
        )
        coordinator_plan = _inject_coordinator_fact_guardrails(coordinator_plan, facts)

        jd_output = await jd_agent.run(
            f"""
{grounding_context}

Coordinator brief:
{coordinator_plan}
"""
        )

        company_output = await company_agent.run(
            f"""
{grounding_context}

Coordinator brief:
{coordinator_plan}

JD analysis:
{jd_output}

Prioritize role-relevant findings and preserve concrete evidence from retrieved results.
"""
        )

        resume_output = await resume_agent.run(
            f"""
{grounding_context}

Coordinator brief:
{coordinator_plan}

JD analysis:
{jd_output}

Company research:
{company_output}

Use provided candidate background directly; avoid hypothetical "if your resume..." guidance when details exist.
"""
        )

        interview_output = await interview_agent.run(
            f"""
{grounding_context}

Coordinator brief:
{coordinator_plan}

JD analysis:
{jd_output}

Company research:
{company_output}

Resume optimization:
{resume_output}

Tailor questions to this role/company context and provided candidate background.
"""
        )

        review_output = await review_agent.run(
            f"""
{grounding_context}

Review this draft package with strict grounding QA. Flag generic statements, unsupported claims, and specificity loss.

[Coordinator Brief]
{coordinator_plan}

[JD Analysis]
{jd_output}

[Company Research]
{company_output}

[Resume Optimization]
{resume_output}

[Interview Preparation]
{interview_output}
"""
        )

        report = await report_agent.run(
            f"""
Create the final structured JobPilot report from these materials. Preserve key specifics and evidence anchors.

{grounding_context}

[Coordinator Brief]
{coordinator_plan}

[JD Analysis]
{jd_output}

[Company Research]
{company_output}

[Resume Optimization]
{resume_output}

[Interview Preparation]
{interview_output}

[Review Findings]
{review_output}
"""
        )

        return report

    def _require_agent(self, key: str):
        agent = self.get_agent(key)
        if agent is None:
            raise ValueError(f"JobPilotFlow requires agent '{key}'")
        return agent
