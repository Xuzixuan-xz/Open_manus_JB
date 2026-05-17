from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Type, Union

if TYPE_CHECKING:
    from app.agent.base import BaseAgent
    from app.flow.base import BaseFlow


class FlowType(str, Enum):
    PLANNING = "planning"
    JOBPILOT = "jobpilot"


class FlowFactory:
    """Factory for creating different types of flows with support for multiple agents"""

    @staticmethod
    def _resolve_flow_class(flow_type: FlowType) -> Type["BaseFlow"]:
        if flow_type == FlowType.PLANNING:
            from app.flow.planning import PlanningFlow

            return PlanningFlow
        if flow_type == FlowType.JOBPILOT:
            from app.flow.jobpilot import JobPilotFlow

            return JobPilotFlow
        raise ValueError(f"Unknown flow type: {flow_type}")

    @staticmethod
    def create_flow(
        flow_type: FlowType,
        agents: Union["BaseAgent", List["BaseAgent"], Dict[str, "BaseAgent"]],
        **kwargs,
    ) -> "BaseFlow":
        flow_class = FlowFactory._resolve_flow_class(flow_type)
        return flow_class(agents, **kwargs)
