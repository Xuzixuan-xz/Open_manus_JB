"""CoverLetterAgent — generates personalized application materials."""
from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.jobpilot.cover_letter import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.jobpilot.md_exporter import MarkdownExporterTool


class CoverLetterAgent(ToolCallAgent):
    """Generates a complete set of job application materials.

    Output:
        30-second self-introduction (Chinese & English),
        Application email (subject + body),
        Personalized cover letter
    """

    name: str = "CoverLetterAgent"
    description: str = (
        "Generates personalized application materials including bilingual self-introductions, "
        "an application email, and a tailored cover letter aligned with the target role."
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_steps: int = 10
    max_observe: int = 12000

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            MarkdownExporterTool(),
            Terminate(),
        )
    )
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
