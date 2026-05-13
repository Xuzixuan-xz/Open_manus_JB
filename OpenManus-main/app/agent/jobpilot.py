from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.jobpilot import (
    COMPANY_RESEARCH_NEXT_STEP_PROMPT,
    COMPANY_RESEARCH_SYSTEM_PROMPT,
    COORDINATOR_NEXT_STEP_PROMPT,
    COORDINATOR_SYSTEM_PROMPT,
    INTERVIEW_NEXT_STEP_PROMPT,
    INTERVIEW_SYSTEM_PROMPT,
    JD_ANALYSIS_NEXT_STEP_PROMPT,
    JD_ANALYSIS_SYSTEM_PROMPT,
    REPORT_NEXT_STEP_PROMPT,
    REPORT_SYSTEM_PROMPT,
    RESUME_OPTIMIZATION_NEXT_STEP_PROMPT,
    RESUME_OPTIMIZATION_SYSTEM_PROMPT,
    REVIEW_NEXT_STEP_PROMPT,
    REVIEW_SYSTEM_PROMPT,
)
from app.tool import CreateChatCompletion, Terminate, ToolCollection, WebSearch


class CoordinatorAgent(ToolCallAgent):
    name: str = "Coordinator"
    description: str = (
        "Top-level orchestrator for job/internship application tasks that routes and "
        "subdivides work across JobPilot specialist agents"
    )
    system_prompt: str = COORDINATOR_SYSTEM_PROMPT
    next_step_prompt: str = COORDINATOR_NEXT_STEP_PROMPT
    max_steps: int = 8
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(CreateChatCompletion(), Terminate())
    )


class JDAnalysisAgent(ToolCallAgent):
    name: str = "JD_Analysis"
    description: str = "Parses and summarizes job descriptions for application planning"
    system_prompt: str = JD_ANALYSIS_SYSTEM_PROMPT
    next_step_prompt: str = JD_ANALYSIS_NEXT_STEP_PROMPT
    max_steps: int = 8
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(CreateChatCompletion(), Terminate())
    )


class ResumeOptimizationAgent(ToolCallAgent):
    name: str = "Resume_Optimization"
    description: str = (
        "Optimizes resume content against JD requirements with realistic phrasing"
    )
    system_prompt: str = RESUME_OPTIMIZATION_SYSTEM_PROMPT
    next_step_prompt: str = RESUME_OPTIMIZATION_NEXT_STEP_PROMPT
    max_steps: int = 8
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(CreateChatCompletion(), Terminate())
    )


class InterviewAgent(ToolCallAgent):
    name: str = "Interview"
    description: str = (
        "Generates interview questions and answer guidance from JD and resume context"
    )
    system_prompt: str = INTERVIEW_SYSTEM_PROMPT
    next_step_prompt: str = INTERVIEW_NEXT_STEP_PROMPT
    max_steps: int = 8
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(CreateChatCompletion(), Terminate())
    )


class CompanyResearchAgent(ToolCallAgent):
    name: str = "Company_Research"
    description: str = (
        "Researches target company and role-relevant background using retrieval tools"
    )
    system_prompt: str = COMPANY_RESEARCH_SYSTEM_PROMPT
    next_step_prompt: str = COMPANY_RESEARCH_NEXT_STEP_PROMPT
    max_steps: int = 8
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            WebSearch(),
            CreateChatCompletion(),
            Terminate(),
        )
    )


class ReviewAgent(ToolCallAgent):
    name: str = "Review"
    description: str = (
        "Reviews generated outputs for consistency, realism, and content quality"
    )
    system_prompt: str = REVIEW_SYSTEM_PROMPT
    next_step_prompt: str = REVIEW_NEXT_STEP_PROMPT
    max_steps: int = 8
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(CreateChatCompletion(), Terminate())
    )


class ReportAgent(ToolCallAgent):
    name: str = "Report"
    description: str = "Aggregates specialist outputs into a structured JobPilot report"
    system_prompt: str = REPORT_SYSTEM_PROMPT
    next_step_prompt: str = REPORT_NEXT_STEP_PROMPT
    max_steps: int = 8
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(CreateChatCompletion(), Terminate())
    )
