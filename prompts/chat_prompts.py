from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional

from langchain_core.prompts import ChatPromptTemplate

from .system_prompts import (
    USER_BOUNDARY_END,
    USER_BOUNDARY_START,
    build_system_prompt,
)


def build_chat_prompt(
    history: Iterable[Mapping[str, str]],
    user_input: str,
    *,
    context_chunks: Optional[List[str]] = None,
    tool_results: Optional[List[str]] = None,
) -> List[Any]:
    """Build the ReAct prompt as LangChain chat messages."""
    system_prompt = build_system_prompt()

    react_lines: List[str] = []
    react_lines.append(
        "You are a useful assistant. Use the context of the knowledge base (RAG) to respond, if appropriate."
    )
    react_lines.append("You may call tools when necessary.")
    react_lines.append("")
    react_lines.append(
        "ReAct format — output exactly these four lines in order:\n"
        "Thought: <brief reasoning>\n"
        "Action: <tool name or none>\n"
        "Action Input: <one line; can be empty>\n"
        "Final Answer: <short answer in Russian>\n"
        "\n"
        "Rules:\n"
        "- Use a tool only when necessary; otherwise Action: none.\n"
        "- When knowledge base context is provided, ground your final answer in it.\n"
    )

    human_lines: List[str] = []
    if context_chunks:
        human_lines.append("Additional context from knowledge base:")
        for i, chunk in enumerate(context_chunks, start=1):
            human_lines.append(f"[Context {i}] {chunk}")
        human_lines.append("")

    if tool_results:
        human_lines.append("Previous tool observations:")
        for i, result in enumerate(tool_results, start=1):
            human_lines.append(f"Observation {i}: {result}")
        human_lines.append("")

    if history:
        human_lines.append("Conversation so far:")
        for message in history:
            role = message.get("role", "user")
            content = message.get("content", "")
            human_lines.append(f"{role.title()}: {content}")
        human_lines.append("")

    human_lines.append(f"{USER_BOUNDARY_START}")
    human_lines.append(user_input)
    human_lines.append(f"{USER_BOUNDARY_END}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("system", "\n".join(react_lines)),
            ("human", "\n".join(human_lines)),
        ]
    )
    return prompt.format_messages()
