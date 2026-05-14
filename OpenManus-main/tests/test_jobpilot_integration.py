import importlib
from pathlib import Path

import pytest


def _write_test_config() -> tuple[Path, bool]:
    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "config"
    config_file = config_dir / "config.toml"
    if config_file.exists():
        return config_file, False

    config_file.write_text(
        """
[llm]
model = "test-model"
base_url = "https://example.com/v1"
api_key = "test-key"
max_tokens = 1024
temperature = 0.0

[llm.vision]
model = "test-model"
base_url = "https://example.com/v1"
api_key = "test-key"
max_tokens = 1024
temperature = 0.0

[runflow]
use_data_analysis_agent = false
use_jobpilot_flow = false

[daytona]
daytona_api_key = "test-daytona-key"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_file, True


@pytest.fixture(scope="module", autouse=True)
def temp_config_file():
    config_file, created = _write_test_config()
    yield
    if created and config_file.exists():
        config_file.unlink()


def test_jobpilot_agents_registered():
    jobpilot = importlib.import_module("app.agent.jobpilot")

    assert jobpilot.CoordinatorAgent.model_fields["name"].default == "Coordinator"
    assert jobpilot.JDAnalysisAgent.model_fields["name"].default == "JD_Analysis"
    assert (
        jobpilot.ResumeOptimizationAgent.model_fields["name"].default
        == "Resume_Optimization"
    )
    assert jobpilot.InterviewAgent.model_fields["name"].default == "Interview"
    assert (
        jobpilot.CompanyResearchAgent.model_fields["name"].default == "Company_Research"
    )
    assert jobpilot.ReviewAgent.model_fields["name"].default == "Review"
    assert jobpilot.ReportAgent.model_fields["name"].default == "Report"


def test_jobpilot_flow_factory_registration():
    flow_factory_module = importlib.import_module("app.flow.flow_factory")
    jobpilot_flow_module = importlib.import_module("app.flow.jobpilot")

    flow = flow_factory_module.FlowFactory.create_flow(
        flow_type=flow_factory_module.FlowType.JOBPILOT,
        agents={},
    )

    assert isinstance(flow, jobpilot_flow_module.JobPilotFlow)


def test_runflow_settings_default_for_jobpilot():
    config_module = importlib.import_module("app.config")

    runflow_settings = config_module.RunflowSettings()
    assert runflow_settings.use_jobpilot_flow is False


def test_jobpilot_prompts_enforce_grounding():
    prompts = importlib.import_module("app.prompt.jobpilot")

    assert "Reject generic boilerplate plans" in prompts.COORDINATOR_SYSTEM_PROMPT
    assert "Prioritize role-relevant findings" in prompts.COMPANY_RESEARCH_SYSTEM_PROMPT
    assert "Do not default to generic trend searches" in prompts.COMPANY_RESEARCH_SYSTEM_PROMPT
    assert "Search-query constraints:" in prompts.COMPANY_RESEARCH_NEXT_STEP_PROMPT
    assert (
        'avoid hypothetical phrasing like "if your resume includes..."'
        in prompts.RESUME_OPTIMIZATION_SYSTEM_PROMPT
    )
    assert (
        "tailored to the target role/company context" in prompts.INTERVIEW_SYSTEM_PROMPT
    )
    assert "strict QA and grounding auditor" in prompts.REVIEW_SYSTEM_PROMPT
    assert "Preserve high-value specifics" in prompts.REPORT_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_jobpilot_flow_passes_grounding_context():
    flow_module = importlib.import_module("app.flow.jobpilot")

    class StubAgent:
        def __init__(self, response: str):
            self.response = response
            self.calls: list[str] = []

        async def run(self, request: str) -> str:
            self.calls.append(request)
            return self.response

    coordinator = StubAgent("coordinator-plan")
    jd = StubAgent("jd-analysis")
    company = StubAgent("company-research")
    resume = StubAgent("resume-optimization")
    interview = StubAgent("interview-prep")
    review = StubAgent("review-findings")
    report = StubAgent("final-report")

    flow = flow_module.JobPilotFlow.model_construct(
        agents={
            "coordinator": coordinator,
            "jd_analysis": jd,
            "company_research": company,
            "resume_optimization": resume,
            "interview": interview,
            "review": review,
            "report": report,
        },
        primary_agent_key="coordinator",
    )

    result = await flow.execute("Target role request")

    assert result == "final-report"
    assert (
        "confirmed facts, unknowns, and role-specific priorities"
        in coordinator.calls[0]
    )
    assert "Coordinator brief:\ncoordinator-plan" in jd.calls[0]
    assert "JD analysis:\njd-analysis" in company.calls[0]
    assert "Use user-provided role/company/candidate details as primary query-planning truth." in company.calls[0]
    assert "Avoid repetitive generic trend/job-board searches" in company.calls[0]
    assert "Company research:\ncompany-research" in resume.calls[0]
    assert "User request:\nTarget role request" in resume.calls[0]
    assert "Avoid hypothetical phrasing like \"if the candidate has...\"" in resume.calls[0]
    assert "Company research:\ncompany-research" in interview.calls[0]
    assert "Resume optimization:\nresume-optimization" in interview.calls[0]
    assert "[Grounding Context]\nUser request:\nTarget role request" in review.calls[0]
    assert "[Coordinator Brief]\ncoordinator-plan" in review.calls[0]
    assert "[JD Analysis]\njd-analysis" in review.calls[0]
    assert "[User Request]\nTarget role request" in report.calls[0]
    assert "[Review Findings]\nreview-findings" in report.calls[0]
