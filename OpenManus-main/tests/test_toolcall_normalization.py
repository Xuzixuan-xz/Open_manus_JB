import json
import importlib
from pathlib import Path
from types import SimpleNamespace

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

from app.schema import Function, ToolCall
from app.tool.base import BaseTool, ToolResult
from app.tool.tool_collection import ToolCollection


@pytest.fixture(scope="module", autouse=True)
def temp_config_file():
    yield
    if _CONFIG_CREATED and _CONFIG_FILE.exists():
        _CONFIG_FILE.unlink()


class FakeBrowserTool(BaseTool):
    name: str = "browser_use"
    description: str = "fake browser tool for tests"
    parameters: dict = {"type": "object"}

    async def execute(self, action: str, query: str | None = None, **kwargs) -> ToolResult:
        return ToolResult(output=f"browser:{action}:{query}")


class FakeWebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "fake web search tool for tests"
    parameters: dict = {"type": "object"}

    async def execute(self, query: str, **kwargs) -> ToolResult:
        return ToolResult(output=f"search:{query}")


def _build_agent(include_web_search: bool):
    toolcall_module = importlib.import_module("app.agent.toolcall")
    ToolCallAgent = toolcall_module.ToolCallAgent
    schema_module = importlib.import_module("app.schema")

    tools = [FakeBrowserTool()]
    if include_web_search:
        tools.append(FakeWebSearchTool())

    class _Agent(ToolCallAgent):
        available_tools: ToolCollection = Field(
            default_factory=lambda: ToolCollection(*tools)
        )

    return _Agent.model_construct(
        llm=None,
        memory=schema_module.Memory(),
        available_tools=ToolCollection(*tools),
        special_tool_names=[],
    )


def _tool_call(name: str, arguments: dict | str) -> ToolCall:
    raw_args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(id="call_1", function=Function(name=name, arguments=raw_args))


@pytest.mark.asyncio
async def test_reroutes_malformed_browser_use_web_search_payload():
    agent = _build_agent(include_web_search=True)
    result = await agent.execute_tool(
        _tool_call("browser_use", {"web_search": "字节跳动 公司文化 和 价值观"})
    )

    assert "cmd `web_search`" in result
    assert "search:字节跳动 公司文化 和 价值观" in result


@pytest.mark.asyncio
async def test_normalizes_browser_use_web_search_without_web_search_tool():
    agent = _build_agent(include_web_search=False)
    result = await agent.execute_tool(_tool_call("browser_use", {"web_search": "openmanus"}))

    assert "cmd `browser_use`" in result
    assert "browser:web_search:openmanus" in result


@pytest.mark.asyncio
async def test_valid_browser_use_call_still_works():
    agent = _build_agent(include_web_search=False)
    result = await agent.execute_tool(
        _tool_call("browser_use", {"action": "go_to_url", "url": "https://example.com"})
    )

    assert "cmd `browser_use`" in result
    assert "browser:go_to_url:None" in result


@pytest.mark.asyncio
async def test_normalizes_browser_use_url_without_action():
    """browser_use called with only a url arg (no action) should normalize to go_to_url."""
    agent = _build_agent(include_web_search=False)
    result = await agent.execute_tool(
        _tool_call("browser_use", {"url": "https://example.com"})
    )

    assert "cmd `browser_use`" in result
    assert "browser:go_to_url:None" in result


@pytest.mark.asyncio
async def test_content_only_toolcall_agent_response_finishes_without_looping():
    toolcall_module = importlib.import_module("app.agent.toolcall")
    schema_module = importlib.import_module("app.schema")

    class FakeLLM:
        async def ask_tool(self, **kwargs):
            return SimpleNamespace(content="grounded answer", tool_calls=[])

    agent = toolcall_module.ToolCallAgent.model_construct(
        llm=FakeLLM(),
        memory=schema_module.Memory(),
    )

    result = await agent.run("Summarize the request")

    assert result == "Step 1: grounded answer"
