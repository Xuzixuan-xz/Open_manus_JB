from app.agent.base import BaseAgent


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


def __getattr__(name: str):
    if name in {
        "CoordinatorAgent",
        "JDAnalysisAgent",
        "ResumeOptimizationAgent",
        "InterviewAgent",
        "CompanyResearchAgent",
        "ReviewAgent",
        "ReportAgent",
    }:
        from app.agent import jobpilot

        return getattr(jobpilot, name)
    if name == "MCPAgent":
        from app.agent.mcp import MCPAgent

        return MCPAgent
    if name == "ReActAgent":
        from app.agent.react import ReActAgent

        return ReActAgent
    if name == "SWEAgent":
        from app.agent.swe import SWEAgent

        return SWEAgent
    if name == "ToolCallAgent":
        from app.agent.toolcall import ToolCallAgent

        return ToolCallAgent
    if name == "BrowserAgent":
        try:
            from app.agent.browser import BrowserAgent
        except ModuleNotFoundError as exc:  # optional dependency
            raise ModuleNotFoundError(
                "BrowserAgent dependencies are not installed. "
                "Install optional browser/daytona dependencies (for example: "
                "`pip install browser-use playwright`) to use BrowserAgent."
            ) from exc
        return BrowserAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
