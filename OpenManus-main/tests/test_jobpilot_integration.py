import importlib
from pathlib import Path


def _ensure_test_config() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "config"
    config_file = config_dir / "config.toml"
    if config_file.exists():
        return

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


def test_jobpilot_agents_registered():
    _ensure_test_config()
    jobpilot = importlib.import_module("app.agent.jobpilot")

    assert jobpilot.CoordinatorAgent.model_fields["name"].default == "Coordinator"
    assert jobpilot.JDAnalysisAgent.model_fields["name"].default == "JD_Analysis"
    assert (
        jobpilot.ResumeOptimizationAgent.model_fields["name"].default
        == "Resume_Optimization"
    )
    assert jobpilot.InterviewAgent.model_fields["name"].default == "Interview"
    assert (
        jobpilot.CompanyResearchAgent.model_fields["name"].default
        == "Company_Research"
    )
    assert jobpilot.ReviewAgent.model_fields["name"].default == "Review"
    assert jobpilot.ReportAgent.model_fields["name"].default == "Report"


def test_jobpilot_flow_factory_registration():
    _ensure_test_config()
    flow_factory_module = importlib.import_module("app.flow.flow_factory")
    jobpilot_flow_module = importlib.import_module("app.flow.jobpilot")

    flow = flow_factory_module.FlowFactory.create_flow(
        flow_type=flow_factory_module.FlowType.JOBPILOT,
        agents={},
    )

    assert isinstance(flow, jobpilot_flow_module.JobPilotFlow)


def test_runflow_settings_default_for_jobpilot():
    _ensure_test_config()
    config_module = importlib.import_module("app.config")

    runflow_settings = config_module.RunflowSettings()
    assert runflow_settings.use_jobpilot_flow is False
