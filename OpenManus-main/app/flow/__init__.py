from app.flow.flow_factory import FlowFactory, FlowType

__all__ = ["FlowFactory", "FlowType", "PlanningFlow", "JobPilotFlow"]


def __getattr__(name: str):
    if name == "JobPilotFlow":
        from app.flow.jobpilot import JobPilotFlow

        return JobPilotFlow
    if name == "PlanningFlow":
        from app.flow.planning import PlanningFlow

        return PlanningFlow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
