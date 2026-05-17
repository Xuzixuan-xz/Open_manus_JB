"""ResumeOptimizerAgent — analyzes a resume against a JD and suggests improvements."""
from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.jobpilot.resume_optimizer import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.jobpilot.doc_parser import DocParserTool
from app.tool.jobpilot.md_exporter import MarkdownExporterTool
from app.tool.jobpilot.rag_retriever import RAGRetrieverTool


class ResumeOptimizerAgent(ToolCallAgent):
    """Analyzes a candidate's resume against a parsed JD and produces an optimization report.

    Output sections:
        Match Score, Skill Gap Analysis, Keyword Injection,
        Project/Experience Rewrite Suggestions, Summary Rewrite
    """

    name: str = "ResumeOptimizerAgent"
    description: str = (
        "Analyzes a resume against a job description and provides a match score, "
        "skill gap analysis, and concrete rewrite suggestions using the STAR method."
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_steps: int = 10
    max_observe: int = 12000

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            DocParserTool(),
            RAGRetrieverTool(),
            MarkdownExporterTool(),
            Terminate(),
        )
    )
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
