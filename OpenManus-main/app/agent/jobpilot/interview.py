"""InterviewAgent — generates a tailored interview preparation kit."""
from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.jobpilot.interview import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection, WebSearch


class InterviewAgent(ToolCallAgent):
    """Generates a comprehensive interview preparation kit for a specific role and candidate.

    Output sections:
        Technical Questions (with answer frameworks),
        Behavioral Questions (STAR format),
        Questions to Ask the Interviewer
    """

    name: str = "InterviewAgent"
    description: str = (
        "Generates tailored technical and behavioral interview questions with answer frameworks, "
        "and smart questions to ask the interviewer — all calibrated to the specific role and candidate."
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_steps: int = 10
    max_observe: int = 10000

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            WebSearch(),
            Terminate(),
        )
    )
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
