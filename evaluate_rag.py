from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from datasets import Dataset
from ragas import RunConfig, evaluate
from ragas.metrics import (
    answer_correctness as _legacy_answer_correctness,
    answer_relevancy as _legacy_answer_relevancy,
    faithfulness as _legacy_faithfulness,
)
try:
    from ragas.metrics.collections import (
        answer_correctness as _answer_correctness,
        answer_relevancy as _answer_relevancy,
        faithfulness as _faithfulness,
    )
except ImportError:  # pragma: no cover - compatibility with older ragas
    _answer_correctness = None
    _answer_relevancy = None
    _faithfulness = None

from chains import ChatAgent
from llm_wrapper.ollama_wrapper import create_llm
from rag import QdrantRetriever
from tools import ToolExecutor
from langchain_ollama import OllamaEmbeddings

from utils import sanitize_text
from utils import configure_langsmith

@dataclass
class EvalConfig:
    golden_set_path: str = "data/uploads/golden_set.json"
    output_dir: str = "data/uploads"
    top_k: int = 5
    judge_model: str = "mistral:7b"
    judge_provider: str = "ollama"
    sample_ratio: float = 0.3
    sample_seed: int | None = 42
    generator_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_organization: str | None = None
    ragas_timeout_s: int = 0
    ragas_max_retries: int = 2
    ragas_max_workers: int = 8
    qdrant_collection: str = "rag_docs"
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = False
    qdrant_api_key: str | None = None
    qdrant_https: bool = False
    qdrant_embed_model: str = "nomic-embed-text"


RAGAS_METRIC_RENAMES = {
    "answer_correctness": "answer_accuracy",
    "answer_relevancy": "answer_relevance",
}


def _print_progress(current: int, total: int) -> None:
    if total <= 0:
        return
    percent = (current / total) * 100
    print(f"\rОценка: {current}/{total} ({percent:.1f}%)", end="", flush=True)


def _load_config(path: str) -> EvalConfig:
    config = EvalConfig()
    config_path = Path(path)
    if not config_path.exists():
        return config
    data = json.loads(config_path.read_text(encoding="utf-8"))
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Compose-first: allow overriding connection endpoints via env vars.
    # This avoids hardcoding 127.0.0.1 inside containers.
    config.ollama_base_url = os.getenv("OLLAMA_BASE_URL", config.ollama_base_url)
    config.qdrant_host = os.getenv("QDRANT_HOST", config.qdrant_host)
    config.qdrant_port = int(os.getenv("QDRANT_PORT", config.qdrant_port))
    config.qdrant_grpc_port = int(os.getenv("QDRANT_GRPC_PORT", config.qdrant_grpc_port))
    config.qdrant_prefer_grpc = (
        os.getenv("QDRANT_PREFER_GRPC", str(config.qdrant_prefer_grpc)).lower() == "true"
    )
    config.qdrant_embed_model = os.getenv("QDRANT_EMBED_MODEL", config.qdrant_embed_model)
    return config


def _load_golden_set(path: str) -> List[Dict[str, Any]]:
    golden_path = Path(path)
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden set not found: {golden_path}")
    data = json.loads(golden_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Golden set must be a list of objects.")
    return data


def _sample_golden_set(
    golden_set: Sequence[Dict[str, Any]],
    *,
    ratio: float,
    seed: int | None,
) -> List[Dict[str, Any]]:
    if ratio >= 1.0:
        return list(golden_set)
    if ratio <= 0.0:
        return []
    total = len(golden_set)
    target = max(1, int(round(total * ratio)))
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    selected = set(indices[:target])
    return [sample for idx, sample in enumerate(golden_set) if idx in selected]


def _normalize_source_files(source_value: Any) -> List[str]:
    if source_value is None:
        return []
    if isinstance(source_value, list):
        return [str(value) for value in source_value]
    return [str(source_value)]


def _is_negative_sample(sample: Dict[str, Any]) -> bool:
    return bool(sample.get("is_negative", False))


def _extract_doc_source(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    source = metadata.get("source", "")
    return Path(str(source)).name if source else ""


def _compute_retrieval_metrics(
    samples: Sequence[Dict[str, Any]],
    retrieved_docs: Sequence[List[Any]],
    *,
    top_k: int,
) -> Dict[str, Any]:
    per_sample: List[Dict[str, Any]] = []
    total_hits = 0
    total_mrr = 0.0
    total_recall = 0.0

    for sample, docs in zip(samples, retrieved_docs):
        relevant_files = _normalize_source_files(sample.get("source_file"))
        retrieved_sources = [_extract_doc_source(doc) for doc in docs]
        rank = 0
        for idx, source in enumerate(retrieved_sources, start=1):
            if source in relevant_files:
                rank = idx
                break
        hit = 1 if rank > 0 else 0
        mrr = 1.0 / rank if rank > 0 else 0.0
        recall = hit if relevant_files else 0.0

        total_hits += hit
        total_mrr += mrr
        total_recall += recall

        per_sample.append(
            {
                "question": sample.get("input", ""),
                "source_file": relevant_files,
                "retrieved_sources": retrieved_sources,
                "hit_at_k": hit,
                "mrr": mrr,
                "recall_at_k": recall,
                "rank": rank,
            }
        )

    total = max(len(samples), 1)
    summary = {
        "hit@k": total_hits / total,
        "mrr": total_mrr / total,
        "recall@k": total_recall / total,
        "top_k": top_k,
    }
    return {"summary": summary, "per_sample": per_sample}


def _compute_negative_metrics(
    samples: Sequence[Dict[str, Any]],
    answers: Sequence[str],
) -> Dict[str, Any]:
    per_sample: List[Dict[str, Any]] = []
    total = 0
    correct = 0
    for sample, answer in zip(samples, answers):
        if not _is_negative_sample(sample):
            continue
        total += 1
        normalized_answer = str(answer or "").strip()
        is_correct = normalized_answer == "Информация недоступна"
        if is_correct:
            correct += 1
        per_sample.append(
            {
                "question": sample.get("input", ""),
                "answer": normalized_answer,
                "is_correct": is_correct,
            }
        )

    summary = {
        "negative_total": total,
        "negative_accuracy": (correct / total) if total else None,
        "negative_hallucination_rate": (1 - (correct / total)) if total else None,
    }
    return {"summary": summary, "per_sample": per_sample}


def _rename_metric_keys(record: Dict[str, Any]) -> Dict[str, Any]:
    renamed = {}
    for key, value in record.items():
        renamed[RAGAS_METRIC_RENAMES.get(key, key)] = value
    return renamed


def _summarize_ragas_result(result: Any) -> Dict[str, Any]:
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        summary = df.mean(numeric_only=True).to_dict()
        rows = df.to_dict(orient="records")
        summary = _rename_metric_keys(summary)
        rows = [_rename_metric_keys(row) for row in rows]
        return {"summary": summary, "per_sample": rows}
    if isinstance(result, dict):
        return {"summary": _rename_metric_keys(result), "per_sample": []}
    return {"summary": {}, "per_sample": []}


def _init_metric(metric_obj: Any, *, llm: Any, embeddings: Any) -> Any:
    if hasattr(metric_obj, "required_columns") and hasattr(metric_obj, "name"):
        return metric_obj
    if callable(metric_obj):
        try:
            return metric_obj()
        except TypeError:
            try:
                return metric_obj(llm=llm, embeddings=embeddings)
            except TypeError:
                return metric_obj(llm=llm)
    metric_class = getattr(metric_obj, "AnswerCorrectness", None)
    if metric_class is not None:
        return metric_class(llm=llm, embeddings=embeddings)
    metric_class = getattr(metric_obj, "AnswerRelevancy", None)
    if metric_class is not None:
        return metric_class(llm=llm, embeddings=embeddings)
    metric_class = getattr(metric_obj, "Faithfulness", None)
    if metric_class is not None:
        return metric_class(llm=llm, embeddings=embeddings)
    raise TypeError(f"Unsupported metric object: {metric_obj}")


def _get_ragas_metrics(*, llm: Any, embeddings: Any, use_collections: bool) -> List[Any]:
    if use_collections and _answer_correctness and _answer_relevancy and _faithfulness:
        return [
            _init_metric(_answer_correctness, llm=llm, embeddings=embeddings),
            _init_metric(_answer_relevancy, llm=llm, embeddings=embeddings),
            _init_metric(_faithfulness, llm=llm, embeddings=embeddings),
        ]
    return [
        _init_metric(_legacy_answer_correctness, llm=llm, embeddings=embeddings),
        _init_metric(_legacy_answer_relevancy, llm=llm, embeddings=embeddings),
        _init_metric(_legacy_faithfulness, llm=llm, embeddings=embeddings),
    ]


def _create_judge_llm(config: EvalConfig):
    provider = config.judge_provider.lower().strip()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.judge_model,
            api_key=config.openai_api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.openai_base_url or os.getenv("OPENAI_BASE_URL"),
            organization=config.openai_organization or os.getenv("OPENAI_ORG"),
            temperature=0.0,
        )
    if provider == "ollama":
        return create_llm(
            model=config.judge_model,
            base_url=config.ollama_base_url,
            temperature=0.0,
        )
    raise ValueError(f"Unsupported judge_provider: {config.judge_provider}")


def _create_embeddings(config: EvalConfig) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=config.qdrant_embed_model,
        base_url=config.ollama_base_url,
    )


def _wrap_ragas_llm(llm: Any) -> Any:
    try:
        from ragas.llms import LangchainLLM
    except ImportError:
        try:
            from ragas.llms.langchain import LangchainLLM  # type: ignore[import-not-found]
        except ImportError:
            return llm
    return LangchainLLM(llm)


def run_evaluation(config_path: str) -> Dict[str, Any]:
    # Optional: enable LangSmith tracing for the generator/judge LLM calls.
    # Controlled via env vars (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, etc).
    configure_langsmith()

    config = _load_config(config_path)
    golden_set = _load_golden_set(config.golden_set_path)
    eval_set = _sample_golden_set(
        golden_set,
        ratio=config.sample_ratio,
        seed=config.sample_seed,
    )

    retriever = QdrantRetriever(
        collection_name=config.qdrant_collection,
        host=os.getenv("QDRANT_HOST", config.qdrant_host),
        port=int(os.getenv("QDRANT_PORT", config.qdrant_port)),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", config.qdrant_grpc_port)),
        prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", str(config.qdrant_prefer_grpc)).lower()
        == "true",
        api_key=os.getenv("QDRANT_API_KEY", config.qdrant_api_key),
        https=os.getenv("QDRANT_HTTPS", str(config.qdrant_https)).lower() == "true",
        embedding_model=os.getenv("QDRANT_EMBED_MODEL", config.qdrant_embed_model),
        embedding_base_url=os.getenv("OLLAMA_BASE_URL", config.ollama_base_url),
    )

    generator_llm = create_llm(
        model=config.generator_model,
        base_url=config.ollama_base_url,
        temperature=0.0,
    )
    judge_llm = _create_judge_llm(config)

    agent = ChatAgent(
        llm=generator_llm,
        retriever=retriever,
        tools=ToolExecutor(),
    )

    questions: List[str] = []
    answers: List[str] = []
    contexts_list: List[List[str]] = []
    ground_truths: List[str] = []
    retrieved_docs: List[List[Any]] = []
    eval_samples: List[Dict[str, Any]] = []
    agent_debug: List[Dict[str, Any]] = []

    total_samples = len(eval_set)
    for idx, sample in enumerate(eval_set, start=1):
        question = sanitize_text(str(sample.get("input", "")).strip())
        is_negative = _is_negative_sample(sample)
        expected = sanitize_text(str(sample.get("expected_output", "")).strip())
        if is_negative and not expected:
            expected = "Информация недоступна"
        if not question:
            continue

        docs = retriever.retrieve_documents(question, top_k=config.top_k)
        contexts = [doc.page_content for doc in docs]

        agent_result = agent.run_verbose(
            user_input=question,
            history=[],
            use_rag=True,
            use_tools=False,
            force_retrieval=True,
        )
        answer = agent_result.get("final_answer", "")

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(expected)
        retrieved_docs.append(docs)
        eval_samples.append(sample)
        agent_debug.append(
            {
                "question": question,
                "decision": sanitize_text(agent_result.get("decision", "")),
                "tool_action": agent_result.get("tool_action"),
                "use_rag": bool(agent_result.get("use_rag", False)),
                "use_tools": bool(agent_result.get("use_tools", False)),
            }
        )
        _print_progress(idx, total_samples)

    if total_samples:
        print()

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )

    timeout_value = config.ragas_timeout_s
    if timeout_value is not None and timeout_value <= 0:
        timeout_value = None

    embeddings = _create_embeddings(config)
    use_collections = config.judge_provider.lower().strip() == "openai"
    ragas_llm = _wrap_ragas_llm(judge_llm) if use_collections else judge_llm
    ragas_result = evaluate(
        dataset,
        metrics=_get_ragas_metrics(
            llm=ragas_llm,
            embeddings=embeddings,
            use_collections=use_collections,
        ),
        llm=ragas_llm,
        embeddings=embeddings,
        run_config=RunConfig(
            timeout=timeout_value,
            max_retries=config.ragas_max_retries,
            max_workers=config.ragas_max_workers,
        ),
    )

    positive_samples = [
        sample for sample in eval_samples if not _is_negative_sample(sample)
    ]
    positive_docs = [
        docs for sample, docs in zip(eval_samples, retrieved_docs) if not _is_negative_sample(sample)
    ]
    retrieval_metrics = _compute_retrieval_metrics(
        samples=positive_samples,
        retrieved_docs=positive_docs,
        top_k=config.top_k,
    )
    negative_metrics = _compute_negative_metrics(eval_samples, answers)
    ragas_metrics = _summarize_ragas_result(ragas_result)

    output = {
        "config": asdict(config),
        "sampled_count": len(eval_samples),
        "total_count": len(golden_set),
        "retrieval": retrieval_metrics,
        "negatives": negative_metrics,
        "ragas": ragas_metrics,
        "agent": agent_debug,
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rag_eval_results.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG with RAGAS + retrieval metrics.")
    parser.add_argument(
        "--config",
        default="rag_eval_config.json",
        help="Path to JSON config file.",
    )
    args = parser.parse_args()

    results = run_evaluation(args.config)
    retrieval_summary = results.get("retrieval", {}).get("summary", {})
    ragas_summary = results.get("ragas", {}).get("summary", {})
    negatives_summary = results.get("negatives", {}).get("summary", {})

    print("\nRAG evaluation completed.")
    if retrieval_summary:
        print("Retrieval metrics:", retrieval_summary)
    if ragas_summary:
        print("RAGAS metrics:", ragas_summary)
    if negatives_summary:
        print("Negative metrics:", negatives_summary)


if __name__ == "__main__":
    main()
