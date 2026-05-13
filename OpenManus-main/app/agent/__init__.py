from app.agent.base import BaseAgent
from app.agent.jobpilot import (
    CompanyResearchAgent,
    CoordinatorAgent,
    InterviewAgent,
    JDAnalysisAgent,
    ReportAgent,
    ResumeOptimizationAgent,
    ReviewAgent,
)
from app.agent.mcp import MCPAgent
from app.agent.react import ReActAgent
from app.agent.swe import SWEAgent
from app.agent.toolcall import ToolCallAgent

try:
    from app.agent.browser import BrowserAgent
except ModuleNotFoundError:  # optional dependency (e.g., daytona/browser runtime)
    class BrowserAgent:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "BrowserAgent dependencies are not installed. "
                "Install optional browser/daytona dependencies to use BrowserAgent."
            )


__all__ = [
    "BaseAgent",
    "ReActAgent",
    "SWEAgent",
    "ToolCallAgent",
    "MCPAgent",
    "BrowserAgent",
    "CoordinatorAgent",
    "JDAnalysisAgent",
    "ResumeOptimizationAgent",
    "InterviewAgent",
    "CompanyResearchAgent",
    "ReviewAgent",
    "ReportAgent",
]
