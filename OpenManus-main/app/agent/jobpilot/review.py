"""ReviewAgent — audits all application materials for authenticity and coherence."""
from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.jobpilot.review import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection


class ReviewAgent(ToolCallAgent):
    """Reviews all job application materials for authenticity, alignment, and consistency.

    Output sections:
        Authenticity Check, JD-Resume Alignment Check,
        Consistency Check, Improvement Suggestions,
        Final Score and Recommendation
    """

    name: str = "ReviewAgent"
    description: str = (
        "Reviews all application materials for authenticity (no exaggeration), "
        "JD alignment, and internal consistency — then provides a final readiness score."
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_steps: int = 8
    max_observe: int = 15000

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            Terminate(),
        )
    )
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
