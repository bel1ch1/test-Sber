from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, TypedDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from prompts import build_chat_prompt
from rag import BaseRetriever
from tools import ToolExecutor
from utils import sanitize_messages, sanitize_text

try:
    from langsmith import traceable  # type: ignore
except Exception:  # pragma: no cover
    def traceable(*_args: Any, **_kwargs: Any):  # type: ignore
        def _decorator(func: Any) -> Any:
            return func

        return _decorator


class AgentState(TypedDict, total=False):
    user_input: str
    history: List[Mapping[str, str]]
    use_rag: bool
    use_tools: bool
    force_retrieval: bool
    context_chunks: List[str]
    decision: str
    first_response: str
    tool_action: Optional[Dict[str, str]]
    tool_result: Optional[str]
    final_answer: str


class AgentResult(TypedDict, total=False):
    final_answer: str
    decision: str
    tool_action: Optional[Dict[str, str]]
    use_rag: bool
    use_tools: bool
    force_retrieval: bool


class ChatAgent:
    """LangGraph-style agent: decide -> RAG (optional) -> plan -> tool? -> final."""

    DEFAULT_TOP_K = 5

    def __init__(
        self,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        tools: Optional[ToolExecutor] = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.tools = tools or ToolExecutor()
        self._graph = self._build_graph()

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""

        patterns = [
            re.compile(r"(?is)(?:^|\n)\s*final\s*answer\s*(?:[:：\-—]|\s)\s*(.*)$"),
            re.compile(r"(?is)(?:^|\n)\s*final\s*(?:[:：\-—]|\s)\s*(.*)$"),
            re.compile(r"(?is)(?:^|\n)\s*финальн(?:ый|ое)\s*ответ\s*(?:[:：\-—]|\s)\s*(.*)$"),
        ]

        for pat in patterns:
            matches = list(pat.finditer(raw))
            if not matches:
                continue
            m = matches[-1]
            answer = (m.group(1) or "").strip()
            answer = re.split(
                r"(?im)^\s*(thought|action|action input|observation)\s*[:：]",
                answer,
                maxsplit=1,
            )[0].strip()
            if answer:
                return answer

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        filtered: list[str] = []
        for ln in lines:
            if re.match(r"(?i)^(thought|action|action input|observation)\s*[:：]", ln):
                continue
            filtered.append(ln)
        if filtered:
            return "\n".join(filtered).strip()

        return raw

    @staticmethod
    def _parse_action(text: str) -> Optional[Dict[str, str]]:
        lines = [line.strip() for line in text.splitlines()]
        action_name: Optional[str] = None
        action_input: str = ""
        for i, line in enumerate(lines):
            if line.lower().startswith("action:"):
                action_name = line.split(":", 1)[1].strip()
                if i + 1 < len(lines) and lines[i + 1].lower().startswith("action input:"):
                    action_input = lines[i + 1].split(":", 1)[1].strip()
                break

        if not action_name:
            return None

        return {"name": action_name, "input": action_input}

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("decide", self._node_decide)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("plan", self._node_plan)
        graph.add_node("call_tool", self._node_call_tool)
        graph.add_node("finalize", self._node_finalize)
        graph.add_node("finalize_after_tool", self._node_finalize_after_tool)

        graph.add_conditional_edges("decide", self._route_from_decide)
        graph.add_edge("retrieve", "plan")
        graph.add_conditional_edges("plan", self._route_from_plan)
        graph.add_edge("call_tool", "finalize_after_tool")
        graph.add_edge("finalize", END)
        graph.add_edge("finalize_after_tool", END)

        graph.set_entry_point("decide")
        return graph.compile()

    @traceable(name="agent_decide", run_type="chain")
    def _node_decide(self, state: AgentState) -> Dict[str, Any]:
        user_input = sanitize_text(state.get("user_input", ""))
        base_use_rag = bool(state.get("use_rag", True))
        base_use_tools = bool(state.get("use_tools", False))
        force_retrieval = bool(state.get("force_retrieval", False))

        if force_retrieval:
            return {
                "decision": "retrieval-forced",
                "use_rag": True,
                "use_tools": False,
                "force_retrieval": True,
            }

        if not base_use_rag and not base_use_tools:
            return {"decision": "none", "use_rag": False, "use_tools": False}

        tool_names = [getattr(tool, "name", "") for tool in self.tools.list_tools()]
        tool_names = [name for name in tool_names if name]
        tools_hint = ", ".join(tool_names) if tool_names else "нет доступных инструментов"

        decision_prompt = (
            "Ты маршрутизатор запросов. Определи, нужно ли обращаться к retrieval (RAG), "
            "к tools или ни к чему.\n"
            "- Если пользователь просит выполнить конкретное действие (например, узнать текущее "
            "время, проверить загрузку системы), выбирай tools.\n"
            "- Если вопрос про документы/файлы/загруженные данные/внутренние знания, выбирай retrieval.\n"
            "- Если ни то ни другое — none.\n"
            "- Если нужно и то и другое — both.\n"
            "Ответь одним словом: retrieval | tools | both | none."
        )

        messages = [
            SystemMessage(content=decision_prompt),
            HumanMessage(
                content=(
                    f"Запрос пользователя: {user_input}\n"
                    f"Доступные инструменты: {tools_hint}"
                )
            ),
        ]
        raw_decision = self.llm.invoke(messages)
        decision_text = (
            raw_decision
            if isinstance(raw_decision, str)
            else getattr(raw_decision, "content", str(raw_decision))
        )
        decision = sanitize_text(decision_text).strip().lower()

        wants_tools = "tool" in decision
        wants_retrieval = "retriev" in decision or "rag" in decision
        wants_both = "both" in decision or (wants_tools and wants_retrieval)
        wants_none = "none" in decision and not wants_both

        if wants_both:
            use_tools = base_use_tools
            use_rag = base_use_rag
            resolved = "both"
        elif wants_tools and base_use_tools:
            use_tools = True
            use_rag = False
            resolved = "tools"
        elif wants_retrieval and base_use_rag:
            use_tools = False
            use_rag = True
            resolved = "retrieval"
        elif wants_none:
            use_tools = False
            use_rag = False
            resolved = "none"
        else:
            use_tools = base_use_tools
            use_rag = base_use_rag
            resolved = "fallback"

        return {"decision": resolved, "use_tools": use_tools, "use_rag": use_rag}

    def _route_from_decide(self, state: AgentState) -> str:
        if state.get("use_rag", False):
            return "retrieve"
        return "plan"

    @traceable(name="agent_retrieve", run_type="chain")
    def _node_retrieve(self, state: AgentState) -> Dict[str, Any]:
        use_rag = bool(state.get("use_rag", True))
        user_input = sanitize_text(state.get("user_input", ""))

        context_chunks: List[str] = []
        if use_rag:
            context_chunks = self.retriever.retrieve(user_input, top_k=self.DEFAULT_TOP_K)
            context_chunks = [sanitize_text(c) for c in context_chunks]

        return {"context_chunks": context_chunks, "user_input": user_input}

    @traceable(name="agent_plan", run_type="chain")
    def _node_plan(self, state: AgentState) -> Dict[str, Any]:
        use_tools = bool(state.get("use_tools", False))
        prompt_messages = build_chat_prompt(
            history=state.get("history", []),
            user_input=state.get("user_input", ""),
            context_chunks=state.get("context_chunks", []),
            tool_results=None,
            available_tools=self.tools.list_tools() if use_tools else None,
        )
        prompt_messages = sanitize_messages(prompt_messages)

        llm_for_first_call = self.llm.bind_tools(self.tools.list_tools()) if use_tools else self.llm
        first_response_raw: Any = llm_for_first_call.invoke(prompt_messages)
        first_response = (
            first_response_raw
            if isinstance(first_response_raw, str)
            else getattr(first_response_raw, "content", str(first_response_raw))
        )

        tool_action = self._parse_action(str(first_response)) if use_tools else None
        return {"first_response": str(first_response), "tool_action": tool_action}

    def _route_from_plan(self, state: AgentState) -> str:
        if not state.get("use_tools", False):
            return "finalize"
        action = state.get("tool_action")
        if not action:
            return "finalize"
        if action.get("name", "").strip().lower() == "none":
            return "finalize"
        return "call_tool"

    @traceable(name="agent_tool", run_type="tool")
    def _node_call_tool(self, state: AgentState) -> Dict[str, Any]:
        action = state.get("tool_action") or {}
        tool_name = action.get("name", "").strip()
        tool_input = action.get("input", "")

        try:
            tool_result = self.tools.call(tool_name, tool_input)
        except Exception as exc:  # noqa: BLE001
            tool_result = f"Tool call failed for '{tool_name}': {exc}"

        return {"tool_result": sanitize_text(tool_result)}

    @traceable(name="agent_finalize", run_type="chain")
    def _node_finalize(self, state: AgentState) -> Dict[str, Any]:
        first_response = state.get("first_response", "")
        final_answer = self._extract_final_answer(str(first_response))
        return {"final_answer": sanitize_text(final_answer)}

    @traceable(name="agent_finalize_after_tool", run_type="chain")
    def _node_finalize_after_tool(self, state: AgentState) -> Dict[str, Any]:
        tool_result = sanitize_text(state.get("tool_result", ""))
        if tool_result:
            return {"final_answer": tool_result}

        prompt_messages = build_chat_prompt(
            history=state.get("history", []),
            user_input=state.get("user_input", ""),
            context_chunks=state.get("context_chunks", []),
            tool_results=[state.get("tool_result", "")],
            available_tools=self.tools.list_tools() if state.get("use_tools", False) else None,
        )
        prompt_messages = sanitize_messages(prompt_messages)

        final_response_raw: Any = self.llm.invoke(prompt_messages)
        final_response = (
            final_response_raw
            if isinstance(final_response_raw, str)
            else getattr(final_response_raw, "content", str(final_response_raw))
        )

        final_answer = sanitize_text(self._extract_final_answer(str(final_response)))
        return {"final_answer": final_answer}

    @traceable(name="chat_turn", run_type="chain")
    def run(
        self,
        user_input: str,
        history: Iterable[Mapping[str, str]],
        *,
        use_rag: bool = True,
        use_tools: bool = False,
        force_retrieval: bool = False,
    ) -> str:
        initial_state: AgentState = {
            "user_input": sanitize_text(user_input),
            "history": list(history),
            "use_rag": use_rag,
            "use_tools": use_tools,
            "force_retrieval": force_retrieval,
        }
        result = self._graph.invoke(initial_state)
        return sanitize_text(result.get("final_answer", ""))

    @traceable(name="chat_turn_verbose", run_type="chain")
    def run_verbose(
        self,
        user_input: str,
        history: Iterable[Mapping[str, str]],
        *,
        use_rag: bool = True,
        use_tools: bool = False,
        force_retrieval: bool = False,
    ) -> AgentResult:
        initial_state: AgentState = {
            "user_input": sanitize_text(user_input),
            "history": list(history),
            "use_rag": use_rag,
            "use_tools": use_tools,
            "force_retrieval": force_retrieval,
        }
        result = self._graph.invoke(initial_state)
        return {
            "final_answer": sanitize_text(result.get("final_answer", "")),
            "decision": sanitize_text(result.get("decision", "")),
            "tool_action": result.get("tool_action"),
            "use_rag": bool(result.get("use_rag", False)),
            "use_tools": bool(result.get("use_tools", False)),
            "force_retrieval": bool(result.get("force_retrieval", False)),
        }
