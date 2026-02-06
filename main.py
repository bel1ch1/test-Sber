from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List

from chains import ChatChain
from llm_wrapper.ollama_wrapper import create_llm
from rag import NoopRetriever, QdrantRetriever
from tools import ToolExecutor, moscow_time, system_load

DEFAULT_COLLECTION_NAME = "rag_docs"
UPLOADS_DIR = Path("data/uploads")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 64


def create_qdrant_retriever(collection_name: str) -> QdrantRetriever:
    return QdrantRetriever(
        collection_name=collection_name,
        host=os.getenv("QDRANT_HOST", "127.0.0.1"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
        prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true",
        api_key=os.getenv("QDRANT_API_KEY"),
        https=os.getenv("QDRANT_HTTPS", "false").lower() == "true",
        embedding_model=os.getenv("QDRANT_EMBED_MODEL", "nomic-embed-text"),
        embedding_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )


def build_retriever() -> NoopRetriever | QdrantRetriever:
    collection_name = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME)
    return create_qdrant_retriever(collection_name)


def _run_spinner(stop_event: threading.Event, label: str) -> None:
    frames = "|/-\\"
    idx = 0
    while not stop_event.is_set():
        print(f"\r{label} {frames[idx % len(frames)]}", end="", flush=True)
        time.sleep(0.1)
        idx += 1
    print(f"\r{label} ... готово.{' ' * 10}", end="", flush=True)


def start_spinner(label: str) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    thread = threading.Thread(target=_run_spinner, args=(stop_event, label), daemon=True)
    thread.start()
    return stop_event, thread


def ensure_qdrant_retriever(
    retriever: NoopRetriever | QdrantRetriever,
) -> QdrantRetriever | None:
    if isinstance(retriever, QdrantRetriever):
        return retriever
    collection_name = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME)
    try:
        return create_qdrant_retriever(collection_name)
    except RuntimeError as exc:
        print(f"Ошибка настройки Qdrant: {exc}")
        return None


def iter_text_chunks(
    text: str, *, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> Iterable[str]:
    if chunk_size <= 0:
        return []
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)

    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start >= text_length:
            break


def iter_file_chunks(
    file_path: Path, *, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> Iterable[str]:
    if chunk_size <= 0:
        return []
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)

    buffer = ""
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        while True:
            data = handle.read(chunk_size)
            if not data:
                break
            buffer += data
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size].strip()
                if chunk:
                    yield chunk
                buffer = buffer[chunk_size - chunk_overlap :]
        if buffer.strip():
            yield buffer.strip()


def iter_text_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return (path for path in folder.rglob("*") if path.is_file())


def index_documents(
    retriever: QdrantRetriever,
    folder: Path,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> None:
    files = list(iter_text_files(folder))
    if not files:
        print("В папке нет файлов для обработки.")
        return

    txt_files = [path for path in files if path.suffix.lower() == ".txt"]
    skipped_files = [path for path in files if path.suffix.lower() != ".txt"]
    if not txt_files:
        print("Нет .txt файлов для индексации.")
        if skipped_files:
            print("Пропущенные файлы:")
            for path in skipped_files:
                print(f"- {path.name}")
        return

    stop_event, thread = start_spinner("Идет индексация")
    try:
        retriever.ensure_collection()

        total_chunks = 0
        indexed_files = 0
        batch_texts: List[str] = []
        batch_metadata: List[dict] = []
        for file_path in txt_files:
            chunk_index = 0
            for chunk in iter_file_chunks(
                file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            ):
                chunk_index += 1
                batch_texts.append(chunk)
                batch_metadata.append(
                    {
                        "source": str(file_path),
                        "chunk_index": chunk_index,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    }
                )
                if len(batch_texts) >= BATCH_SIZE:
                    retriever.add_texts(batch_texts, metadatas=batch_metadata)
                    total_chunks += len(batch_texts)
                    batch_texts.clear()
                    batch_metadata.clear()
            if chunk_index > 0:
                indexed_files += 1

        if batch_texts:
            retriever.add_texts(batch_texts, metadatas=batch_metadata)
            total_chunks += len(batch_texts)
            batch_texts.clear()
            batch_metadata.clear()

        if total_chunks == 0:
            print("\rНе удалось подготовить чанки для индексации.              ")
            return
    finally:
        stop_event.set()
        thread.join()
        print()

    print(f"Проиндексировано чанков: {total_chunks} (файлов: {indexed_files}).")
    print(f"Пропущено файлов: {len(skipped_files)}.")
    if skipped_files:
        print("Пропущенные файлы:")
        for path in skipped_files:
            print(f"- {path.name}")


def handle_document_ingest(retriever: QdrantRetriever) -> None:
    print("\nДобавление документов в систему")
    print("-------------------------------")
    print(f"Положите ваши файлы в папку: {UPLOADS_DIR.resolve()}")
    print("Поддерживаемые форматы: .txt, .md")
    print("После этого введите 'upload' для индексации или 'back' для возврата в меню.\n")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        command = input("Команда (upload/back): ").strip().lower()
        if command == "back":
            return
        if command == "upload":
            index_documents(retriever, UPLOADS_DIR)
            return
        print("Неизвестная команда. Используйте 'upload' или 'back'.")


def handle_document_delete(retriever: QdrantRetriever) -> None:
    print("\nУдаление всех документов из системы")
    print("----------------------------------")
    print("Это действие удалит все данные из коллекции Qdrant.")
    print("Локальные файлы в папке uploads останутся без изменений.")
    confirm = input("Подтвердите удаление (yes/no): ").strip().lower()
    if confirm not in {"yes", "y", "да"}:
        print("Удаление отменено.")
        return

    stop_event, thread = start_spinner("Удаление документов")
    try:
        removed = retriever.clear_collection()
    finally:
        stop_event.set()
        thread.join()
        print()
    if removed:
        print("Коллекция очищена. Все документы удалены из векторной базы.")
    else:
        print("Коллекция не найдена. Удалять нечего.")


def show_main_menu() -> str:
    print("\nВыберите действие:")
    print("1) Чат с ассистентом")
    print("2) Добавить документы в систему")
    print("3) Удалить все документы из системы")
    print("4) Выход")
    return input("Введите номер: ").strip()


def main() -> None:
    """Run the console chat: LLM from llm_wrapper, NoopRetriever for RAG, tools via ToolExecutor."""
    llm = create_llm()
    retriever = build_retriever()
    tools = ToolExecutor(tools=[system_load, moscow_time])
    chat_chain = ChatChain(llm=llm, retriever=retriever, tools=tools)

    history: List[Dict[str, str]] = []

    while True:
        choice = show_main_menu()
        if choice == "4":
            print("Пока!")
            break
        if choice == "2":
            qdrant_retriever = ensure_qdrant_retriever(retriever)
            if qdrant_retriever is None:
                continue
            retriever = qdrant_retriever
            chat_chain.retriever = retriever
            handle_document_ingest(qdrant_retriever)
            continue
        if choice == "3":
            qdrant_retriever = ensure_qdrant_retriever(retriever)
            if qdrant_retriever is None:
                continue
            retriever = qdrant_retriever
            chat_chain.retriever = retriever
            handle_document_delete(qdrant_retriever)
            continue
        if choice != "1":
            print("Неизвестный вариант. Выберите 1, 2, 3 или 4.")
            continue

        print("\nКонсольный чат с локальной LLM (RAG + Tools архитектура).")
        print("Введите сообщение. Для выхода наберите 'exit' или 'quit'.\n")
        while True:
            try:
                user_input = input("Вы: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nЗавершение диалога.")
                return

            if user_input.lower() in {"exit", "quit"}:
                print("Возврат в меню.")
                break

            if not user_input:
                continue

            history.append({"role": "user", "content": user_input})

            try:
                assistant_reply = chat_chain.run(
                    user_input=user_input,
                    history=history,
                    use_rag=True,
                    use_tools=True,
                )
            except Exception as exc:  # noqa: BLE001
                assistant_reply = f"Ошибка при обращении к модели: {exc}"

            history.append({"role": "assistant", "content": assistant_reply})
            print(f"Модель: {assistant_reply}\n")


if __name__ == "__main__":
    main()
