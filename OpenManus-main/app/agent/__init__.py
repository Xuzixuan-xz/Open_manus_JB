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
    BrowserAgent = None


__all__ = [
    "BaseAgent",
    "ReActAgent",
    "SWEAgent",
    "ToolCallAgent",
    "MCPAgent",
    "CoordinatorAgent",
    "JDAnalysisAgent",
    "ResumeOptimizationAgent",
    "InterviewAgent",
    "CompanyResearchAgent",
    "ReviewAgent",
    "ReportAgent",
]

if BrowserAgent is not None:
    __all__.append("BrowserAgent")
