from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

try:
    from langsmith import traceable  # type: ignore
except Exception:  # pragma: no cover
    def traceable(*_args: Any, **_kwargs: Any):  # type: ignore
        def _decorator(func: Any) -> Any:
            return func

        return _decorator

Tool = BaseTool


def _tool_input_dict(tool: BaseTool, raw_input: str) -> Dict[str, Any]:
    """Build input dict for BaseTool.invoke() from the tool's schema."""
    try:
        schema = tool.get_input_schema()
        if hasattr(schema, "model_fields"):
            keys = list(schema.model_fields.keys())
        elif hasattr(schema, "__fields__"):
            keys = list(schema.__fields__.keys())
        else:
            keys = ["query"]
        if keys:
            return {keys[0]: raw_input}
    except Exception:
        pass
    return {"query": raw_input}


class ToolExecutor:
    """Registers and runs LangChain BaseTool instances by name via .invoke()."""

    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def call(self, name: str, tool_input: str = "", **kwargs: Any) -> Any:
        """Run the named tool with the given input; uses .invoke() under the hood."""
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' is not registered")
        raw = kwargs.get("query", tool_input) if kwargs else tool_input
        if isinstance(raw, str):
            inp = _tool_input_dict(tool, raw)
        else:
            inp = raw if isinstance(raw, dict) else {"query": str(raw)}

        @traceable(name=f"tool:{name}", run_type="tool")
        def _invoke() -> Any:
            return tool.invoke(inp)

        return _invoke()
