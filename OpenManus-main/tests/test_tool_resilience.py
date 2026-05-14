import importlib
import json
from pathlib import Path

import pytest
from pydantic import Field


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


_CONFIG_FILE, _CONFIG_CREATED = _write_test_config()

from app.exceptions import ToolError
from app.schema import Function, ToolCall
from app.tool.base import BaseTool, ToolResult
from app.tool.file_operators import LocalFileOperator
from app.tool.tool_collection import ToolCollection


@pytest.fixture(scope="module", autouse=True)
def temp_config_file():
    yield
    if _CONFIG_CREATED and _CONFIG_FILE.exists():
        _CONFIG_FILE.unlink()


@pytest.mark.asyncio
async def test_local_file_operator_creates_missing_parent_directories(tmp_path):
    operator = LocalFileOperator()
    missing_parent_file = tmp_path / "workspace" / "resume.txt"

    await operator.write_file(missing_parent_file, "hello")

    assert missing_parent_file.exists()
    assert missing_parent_file.read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_local_file_operator_reports_non_utf8_file_as_tool_error(tmp_path):
    operator = LocalFileOperator()
    binary_file = tmp_path / "binary.dat"
    binary_file.write_bytes(b"\xb2\x00\xff")

    with pytest.raises(ToolError, match="not valid UTF-8 text"):
        await operator.read_file(binary_file)


class FakeFailingTool(BaseTool):
    name: str = "always_fail"
    description: str = "returns a tool error"
    parameters: dict = {"type": "object"}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(error="simulated failure")


def _build_failure_agent():
    toolcall_module = importlib.import_module("app.agent.toolcall")
    ToolCallAgent = toolcall_module.ToolCallAgent
    schema_module = importlib.import_module("app.schema")

    class _Agent(ToolCallAgent):
        available_tools: ToolCollection = Field(
            default_factory=lambda: ToolCollection(FakeFailingTool())
        )

    return _Agent.model_construct(
        llm=None,
        memory=schema_module.Memory(),
        available_tools=ToolCollection(FakeFailingTool()),
        special_tool_names=[],
    )


def _tool_call(name: str, arguments: dict | str) -> ToolCall:
    raw_args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(id="call_1", function=Function(name=name, arguments=raw_args))


@pytest.mark.asyncio
async def test_repeated_failed_tool_call_adds_recovery_hint():
    agent = _build_failure_agent()
    call = _tool_call("always_fail", {"path": "/tmp/missing.txt"})

    first_result = await agent.execute_tool(call)
    second_result = await agent.execute_tool(call)

    assert "simulated failure" in first_result
    assert "failed multiple times" in second_result
