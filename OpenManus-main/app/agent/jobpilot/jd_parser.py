"""JDParserAgent — parses job descriptions into structured JSON."""
from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.jobpilot.jd_parser import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection, WebSearch
from app.tool.jobpilot.web_scraper import WebScraperTool


class JDParserAgent(ToolCallAgent):
    """Parses a job description (raw text or URL) into structured JSON.

    Output JSON fields:
        required_skills, nice_to_have, responsibilities,
        culture_keywords, seniority
    """

    name: str = "JDParserAgent"
    description: str = (
        "Parses a job description into structured JSON: required skills, "
        "nice-to-have skills, responsibilities, culture keywords, and seniority level."
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_steps: int = 10
    max_observe: int = 8000

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            WebScraperTool(),
            WebSearch(),
            Terminate(),
        )
    )
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
