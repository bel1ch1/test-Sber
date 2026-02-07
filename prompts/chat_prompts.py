from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional

from langchain_core.prompts import ChatPromptTemplate

from utils import sanitize_text

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
    available_tools: Optional[Iterable[Any]] = None,
) -> List[Any]:
    """Build the ReAct prompt as LangChain chat messages."""
    system_prompt = sanitize_text(build_system_prompt())

    react_lines: List[str] = []
    react_lines.append(
        "You are a useful assistant. If the user asks a specific question that depends on local knowledge, "
        "answer strictly using the knowledge base (RAG) context."
    )
    react_lines.append(
        "Treat a question as 'local knowledge' if it refers to documents, files, records, internal policies, "
        "uploaded data, or any facts that could only come from the knowledge base."
    )
    react_lines.append(
        "If the knowledge base context does not contain the answer to such a question, reply: "
        '"Информация недоступна".'
    )
    react_lines.append(
        "For general conversation or questions not dependent on local knowledge, respond normally."
    )
    react_lines.append("You may call tools when necessary.")
    react_lines.append("")

    if available_tools:
        tool_lines: List[str] = []
        for tool in available_tools:
            name = getattr(tool, "name", None)
            desc = getattr(tool, "description", "") or ""
            short_desc = desc.splitlines()[0].strip() if isinstance(desc, str) else ""
            if name:
                tool_lines.append(f"- {sanitize_text(str(name))}: {sanitize_text(short_desc)}")
        if tool_lines:
            react_lines.append("Available tools (use EXACT name in Action):")
            react_lines.extend(tool_lines)
            react_lines.append("")

    react_lines.append(
        "Few-shot examples (follow the pattern; do NOT copy verbatim):\n"
        "Example A — ask Moscow time:\n"
        "Thought: Нужно узнать текущее время в Москве.\n"
        "Action: moscow_time\n"
        "Action Input:\n"
        "Final Answer: Сейчас уточню время в Москве.\n"
        "\n"
        "Example B — ask about CPU/RAM load:\n"
        "Thought: Нужно проверить текущую загрузку CPU и RAM.\n"
        "Action: system_load\n"
        "Action Input:\n"
        "Final Answer: Сейчас проверю нагрузку системы.\n"
        "\n"
        "Example C — no tool needed:\n"
        "Thought: Инструменты не нужны.\n"
        "Action: none\n"
        "Action Input:\n"
        "Final Answer: ..."
    )
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
        "- If tool observations are provided, your Final Answer MUST include them directly. "
        "Do not say you will check or are about to check; provide the result.\n"
    )

    human_lines: List[str] = []
    if context_chunks:
        human_lines.append("Additional context from knowledge base:")
        for i, chunk in enumerate(context_chunks, start=1):
            human_lines.append(f"[Context {i}] {sanitize_text(chunk)}")
        human_lines.append("")

    if tool_results:
        human_lines.append("Previous tool observations:")
        for i, result in enumerate(tool_results, start=1):
            human_lines.append(f"Observation {i}: {sanitize_text(result)}")
        human_lines.append("")

    if history:
        human_lines.append("Conversation so far:")
        for message in history:
            role = message.get("role", "user")
            content = message.get("content", "")
            human_lines.append(f"{sanitize_text(role).title()}: {sanitize_text(content)}")
        human_lines.append("")

    human_lines.append(f"{USER_BOUNDARY_START}")
    human_lines.append(sanitize_text(user_input))
    human_lines.append(f"{USER_BOUNDARY_END}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("system", "\n".join(sanitize_text(line) for line in react_lines)),
            ("human", "\n".join(human_lines)),
        ]
    )
    return prompt.format_messages()
