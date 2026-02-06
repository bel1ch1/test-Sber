from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models


class BaseRetriever(ABC):
    """Interface for RAG retrievers (e.g. vector store, DB, files)."""

    @abstractmethod
    def retrieve(self, query: str, *, top_k: int = 5) -> List[str]:
        """Return relevant text chunks for the query."""
        raise NotImplementedError

    def retrieve_documents(self, query: str, *, top_k: int = 5) -> List["Document"]:
        """Return retrieved documents with metadata when available."""
        return []


class NoopRetriever(BaseRetriever):
    """Retriever that returns no chunks; use until a real RAG backend is wired in."""

    def retrieve(self, query: str, *, top_k: int = 5) -> List[str]:
        return []

    def retrieve_documents(self, query: str, *, top_k: int = 5) -> List["Document"]:
        return []


class QdrantRetriever(BaseRetriever):
    """Qdrant-based retriever using LangChain's Qdrant wrapper."""

    def __init__(
        self,
        *,
        collection_name: str,
        host: str = "127.0.0.1",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        api_key: Optional[str] = None,
        https: bool = False,
        timeout: float = 10.0,
        embedding_model: str = "nomic-embed-text",
        embedding_base_url: str = "http://127.0.0.1:11434",
        embedding: Optional["OllamaEmbeddings"] = None,
        create_collection_if_missing: bool = True,
        use_mmr: bool = False,
        mmr_fetch_k: Optional[int] = None,
    ) -> None:
        from langchain_ollama import OllamaEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        self.embedding = embedding or OllamaEmbeddings(
            model=embedding_model,
            base_url=embedding_base_url,
        )
        self.use_mmr = use_mmr
        self.mmr_fetch_k = mmr_fetch_k
        self.client = QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            api_key=api_key,
            https=https,
            timeout=timeout,
        )
        if create_collection_if_missing:
            self.ensure_collection()
        self.vector_store: "QdrantVectorStore" = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embedding,
        )

    def retrieve_documents(self, query: str, *, top_k: int = 5) -> List["Document"]:
        if self.use_mmr:
            fetch_k = self.mmr_fetch_k or max(top_k * 4, top_k)
            return self.vector_store.max_marginal_relevance_search(
                query,
                k=top_k,
                fetch_k=fetch_k,
            )
        return self.vector_store.similarity_search(query, k=top_k)

    def retrieve(self, query: str, *, top_k: int = 5) -> List[str]:
        docs = self.retrieve_documents(query, top_k=top_k)
        return [doc.page_content for doc in docs]

    def add_texts(
        self,
        texts: Sequence[str],
        *,
        metadatas: Optional[Sequence[dict]] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        return self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def add_documents(self, documents: Iterable, *, ids: Optional[Sequence[str]] = None) -> List[str]:
        return self.vector_store.add_documents(documents=documents, ids=ids)

    def ensure_collection(self) -> None:
        from qdrant_client.http import models as qdrant_models

        if self.client.collection_exists(self.collection_name):
            return
        try:
            vector_size = len(self.embedding.embed_query("init"))
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "nomic-embed-text" in message and "pull" in message:
                raise RuntimeError(
                    "Не найден embedding-модель 'nomic-embed-text' в Ollama. "
                    "Выполните: `ollama pull nomic-embed-text` "
                    "или задайте переменную QDRANT_EMBED_MODEL."
                ) from exc
            raise
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    def clear_collection(self) -> bool:
        if not self.client.collection_exists(self.collection_name):
            return False
        self.client.delete_collection(self.collection_name)
        self.ensure_collection()
        return True

    def close(self) -> None:
        self.client.close()
