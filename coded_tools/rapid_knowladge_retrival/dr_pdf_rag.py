"""Vector PDF RAG tool for rapid_knowladge_retrival.

This tool mirrors the behavior expected by the
`rapid_knowladge_retrival` registry entry named `dr_pdf_rag`.
It uses the shared `BaseRag` infrastructure for embedding and retrieval
while enforcing the sentinel fallback message that the network
orchestration depends upon (must start with the exact phrase
`No relevant info found` when no data is available).

Args passed in by the registry (example):
    {
        "directory": "coded_tools/rapid_knowladge_retrival/pdf_corpus",
        "save_vector_store": true,
        "vector_store_path": "coded_tools/rapid_knowladge_retrival/vector_store.json",
        "vector_store_type": "in_memory"
    }

Environment overrides (optional):
    POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_HOST /
    POSTGRES_PORT / POSTGRES_DB (only if vector_store_type = "postgres")

    DISASTER_RAG_EMBED_MODEL  -> custom OpenAI embedding model name

Sentinel fallback string (for manager routing):
    "No relevant info found in local disaster recovery PDFs ..."

The first three words MUST remain: "No relevant info" for routing logic.
"""
from __future__ import annotations

import os
import glob
import logging
from typing import Any, Dict, List, Optional

try:  # Dependency guard for PDF loading
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError:  # pragma: no cover
    PyMuPDFLoader = None  # type: ignore

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from neuro_san.interfaces.coded_tool import CodedTool
from coded_tools.base_rag import BaseRag, PostgresConfig

logger = logging.getLogger(__name__)

SENTINEL_PREFIX = "No relevant info found"
DEFAULT_DIR = "coded_tools/rapid_knowladge_retrival/pdf_corpus"


def get_tool():  # pragma: no cover - factory for toolbox loading
    return DrPdfRag()


class DrPdfRag(CodedTool, BaseRag):
    """RAG over disaster recovery PDFs for rapid_knowladge_retrival.

    Provides vector store creation (in-memory or Postgres) with optional JSON
    persistence for in-memory mode. Returns top-k matches with source + page
    metadata. Falls back to sentinel string if no documents or results.
    """

    def __init__(self) -> None:  # noqa: D401
        super().__init__()
        embed_model = os.getenv("DISASTER_RAG_EMBED_MODEL") or None
        if embed_model:
            self.embeddings = OpenAIEmbeddings(model=embed_model)

    async def async_invoke(
        self,
        args: Dict[str, Any],
        sly_data: Dict[str, Any],  # pylint: disable=unused-argument
    ) -> str:
        query: str = str(args.get("query", "")).strip()
        if not query:
            return "❌ Missing required input: 'query'."

        directory: str = str(args.get("directory", DEFAULT_DIR))
        vector_store_type: str = str(
            args.get("vector_store_type", "in_memory")
        )
        self.save_vector_store = bool(args.get("save_vector_store", False))

        abs_directory = directory
        if not os.path.isabs(abs_directory):
            # Get the project root directory (contains coded_tools)
            base_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            abs_directory = os.path.abspath(
                os.path.join(base_root, abs_directory)
            )

        vector_store_path: Optional[str] = (
            args.get("vector_store_path")
            or ("vector_store.json" if self.save_vector_store else None)
        )
        if vector_store_path and not os.path.isabs(vector_store_path):
            # Get the rapid_knowladge_retrival directory path
            # Get parent of pdf_corpus directory
            base_dir = os.path.dirname(abs_directory)
            vector_store_path = os.path.join(base_dir, vector_store_path)
        self.configure_vector_store_path(vector_store_path)

        postgres_config: Optional[PostgresConfig] = None
        table_name: Optional[str] = args.get("table_name")
        if vector_store_type == "postgres":
            if not table_name:
                return (
                    "❌ Missing required input: 'table_name' for postgres "
                    "backend."
                )
            postgres_config = PostgresConfig(
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                host=os.getenv("POSTGRES_HOST"),
                port=os.getenv("POSTGRES_PORT"),
                database=os.getenv("POSTGRES_DB"),
                table_name=table_name,
            )

        try:
            logger.info(
                "Creating vector store: dir=%s, path=%s",
                directory,
                vector_store_path
            )
            vector_store = await self.generate_vector_store(
                loader_args={
                    "directory": directory,
                    "glob_pattern": "*.pdf",
                },
                postgres_config=postgres_config,
                vector_store_type=vector_store_type,
            )
        except ValueError as ve:
            logger.error("Configuration error: %s", ve)
            return f"❌ Configuration error: {ve}"
        except Exception as exc:  # pragma: no cover
            # Broad catch guarded by logging; unexpected infrastructure error.
            logger.exception("Unexpected failure building vector store")
            return f"❌ Failed to build vector store: {exc}"

        if not vector_store:
            return (
                f"{SENTINEL_PREFIX} in local disaster recovery PDFs "
                "(no vector store). Ensure PDFs exist in "
                "coded_tools/rapid_knowladge_retrival/pdf_corpus/."
            )

        baseline: str = await self.query_vectorstore(vector_store, query)
        if not baseline.strip():
            return (
                f"{SENTINEL_PREFIX} in local disaster recovery PDFs "
                "(empty retrieval)."
            )

        try:
            retriever = vector_store.as_retriever(
                search_kwargs={"k": int(args.get("top_k", 5))}
            )
            docs: List[Document] = await retriever.ainvoke(query)
        except Exception as exc:
            # Broad catch: retriever construction may raise from backend libs.
            logger.warning(
                "Refined retrieval failed: %s; returning baseline", exc
            )
            return baseline

        if not docs:
            return (
                f"{SENTINEL_PREFIX} in local disaster recovery PDFs "
                "(no documents matched)."
            )

        formatted: List[str] = []
        references: List[str] = []
        for i, doc in enumerate(docs, start=1):
            src = (
                doc.metadata.get("source")
                or doc.metadata.get("file_path")
                or "unknown_source"
            )
            page = doc.metadata.get("page_number") or doc.metadata.get("page")
            base_src = os.path.basename(str(src))
            header = f"[Result {i}] Source: {base_src}"
            if page is not None:
                header += f", Page: {page}"
                references.append(f"- {base_src} (p. {page})")
            else:
                references.append(f"- {base_src}")
            formatted.append(f"{header}\n{doc.page_content.strip()}")

        seen = set()
        unique_refs: List[str] = []
        for ref in references:
            if ref not in seen:
                seen.add(ref)
                unique_refs.append(ref)

        refs_block = ""
        if unique_refs:
            refs_block = ("\n\nReferences:\n" + "\n".join(unique_refs))
        return "\n\n".join(formatted) + refs_block

    async def load_documents(
        self, loader_args: Dict[str, Any]
    ) -> List[Document]:
        """Load PDFs from the specified directory using PyMuPDFLoader."""
        if PyMuPDFLoader is None:
            logger.warning(
                "PyMuPDFLoader unavailable; returning empty document list."
            )
            return []

        directory: str = loader_args.get("directory", DEFAULT_DIR)
        glob_pattern: str = loader_args.get("glob_pattern", "*.pdf")

        if not os.path.isabs(directory):
            # Get the project root directory (contains coded_tools)
            base_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            candidate = os.path.abspath(os.path.join(base_root, directory))
            directory = candidate

        if not os.path.isdir(directory):
            logger.info(
                "Directory '%s' not found for dr_pdf_rag; returning no docs.",
                directory,
            )
            return []

        pdf_files: List[str] = glob.glob(os.path.join(directory, glob_pattern))
        if not pdf_files:
            logger.info(
                "No PDF files matched pattern '%s' in '%s'.",
                glob_pattern,
                directory,
            )
            return []

        docs: List[Document] = []
        for path in pdf_files:
            try:
                loader = PyMuPDFLoader(file_path=path)
                loaded: List[Document] = await loader.aload()
                docs.extend(loaded)
                logger.info("Loaded PDF: %s (%d pages)", path, len(loaded))
            except Exception as exc:  # pragma: no cover
                # Individual PDF failures logged and skipped.
                logger.error("Failed to load %s: %s", path, exc)
        return docs


__all__ = ["DrPdfRag", "get_tool"]

