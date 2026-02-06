from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from langchain_core.language_models import BaseLanguageModel

from rag import BaseRetriever
from tools import ToolExecutor
from prompts import build_chat_prompt
from prompts.system_prompts import (
    USER_BOUNDARY_END,
    USER_BOUNDARY_START,
    build_system_prompt,
)


class ChatChain:
    """Runs one turn of dialogue: RAG retrieval, ReAct prompt, optional tool call, then final answer."""

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

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        """Extract the part after 'Final Answer:' from the model output, or return the whole text."""
        marker_variants = ["Final Answer:", "Final answer:", "Final answer :"]
        for marker in marker_variants:
            if marker in text:
                return text.split(marker, 1)[1].strip()
        return text.strip()

    @staticmethod
    def _parse_action(text: str) -> Optional[Dict[str, str]]:
        """Parse Action and Action Input from the model's ReAct block."""
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

    def run(
        self,
        user_input: str,
        history: Iterable[Mapping[str, str]],
        *,
        use_rag: bool = True,
        use_tools: bool = False,
    ) -> str:
        """Run one dialogue turn: RAG, ReAct, optional tool call, then reply."""
        # RAG
        context_chunks: List[str] = []
        if use_rag:
            context_chunks = self.retriever.retrieve(user_input, top_k=self.DEFAULT_TOP_K)

        # Build ReAct prompt
        prompt_messages = build_chat_prompt(
            history=history,
            user_input=user_input,
            context_chunks=context_chunks,
            tool_results=None,
        )

        # First LLM call: plan and optionally choose a tool
        llm_for_first_call = (
            self.llm.bind_tools(self.tools.list_tools()) if use_tools else self.llm
        )
        first_response_raw: Any = llm_for_first_call.invoke(prompt_messages)
        first_response = (
            first_response_raw
            if isinstance(first_response_raw, str)
            else getattr(first_response_raw, "content", str(first_response_raw))
        )

        if not use_tools:
            return self._extract_final_answer(str(first_response))

        # Parse Action from model output
        action = self._parse_action(str(first_response))
        if not action:
            return self._extract_final_answer(str(first_response))

        tool_name = action["name"].strip()
        tool_input = action["input"]

        # "none" means no tool call
        if tool_name.lower() == "none":
            return self._extract_final_answer(str(first_response))

        try:
            tool_result = self.tools.call(tool_name, tool_input)
        except Exception as exc:  # noqa: BLE001
            tool_result = f"Tool call failed for '{tool_name}': {exc}"

        # Second LLM call: turn tool result into a final answer
        final_prompt = build_chat_prompt(
            history=history,
            user_input=user_input,
            context_chunks=context_chunks,
            tool_results=[str(tool_result)],
        )

        final_response_raw: Any = self.llm.invoke(final_prompt)
        final_response = (
            final_response_raw
            if isinstance(final_response_raw, str)
            else getattr(final_response_raw, "content", str(final_response_raw))
        )

        return self._extract_final_answer(str(final_response))
