"""Lightweight RAG for the agent platform.

Design goals for this prototype:
- Work offline with no mandatory embedding API.
- Support common file types: txt/md/csv/json + optional pdf/docx.
- Store indexes under AGENT_FILES_DIR/.rag/<index_name>/
- Retrieve top-k chunks by cosine similarity over bag-of-words vectors.

This is intentionally simple so local LLMs only see retrieved chunks,
not entire files.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from backend.compiler.executors import NodeExecutor, exec_generic_tool
from backend.models.agent_spec import AgentNode, AgentSpecification
from backend.real_executors.tools import ConnectedExecutionError, _resolve_scoped_path


def _files_base() -> Path:
    base = Path(os.environ.get("AGENT_FILES_DIR", "agent_platform_files"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _index_dir(index_name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", index_name or "default").strip("._") or "default"
    d = _files_base() / ".rag" / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


_TOKEN = re.compile(r"[A-Za-z0-9_]{2,}")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def _vectorize(text: str) -> dict[str, float]:
    toks = _tokenize(text)
    if not toks:
        return {}
    counts = Counter(toks)
    n = float(len(toks))
    return {t: c / n for t, c in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        piece = text[i : i + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if i + chunk_size >= n:
            break
        i += max(1, chunk_size - overlap)
    return chunks


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".log", ".csv", ".json", ".py", ".ts", ".tsx", ".js"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
            df = pd.read_excel(path)
            return df.to_csv(index=False)
        except Exception as e:
            raise ConnectedExecutionError(f"Could not read spreadsheet '{path.name}': {e}") from e
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            parts = [(page.extract_text() or "") for page in reader.pages]
            text_out = "\n".join(parts).strip()
            if not text_out:
                raise ConnectedExecutionError(
                    "PDF '" + path.name + "' has no extractable text (likely scanned/image-only). "
                    "Convert to a text PDF or .txt, or OCR it first."
                )
            return text_out
        except ImportError as e:
            raise ConnectedExecutionError("PDF support needs pypdf. Run: pip install pypdf") from e
        except ConnectedExecutionError:
            raise
        except Exception as e:
            raise ConnectedExecutionError("Could not read PDF '" + path.name + "': " + str(e)) from e

    if suffix == ".docx":
        try:
            import docx
            doc = docx.Document(str(path))
            return "\n".join(para.text for para in doc.paragraphs)
        except ImportError as e:
            raise ConnectedExecutionError(
                "DOCX support needs the 'python-docx' package. Install it or save as .txt."
            ) from e
        except Exception as e:
            raise ConnectedExecutionError(f"Could not read DOCX '{path.name}': {e}") from e
    if suffix in IMAGE_EXTENSIONS:
        # Images are not OCR'd here — they are stored as image refs for vision models.
        return f"[image:{path.name}]"
    raise ConnectedExecutionError(
        f"Unsupported file type '{suffix}' for RAG. Use txt, md, csv, json, pdf, docx, xlsx, or images (png/jpg/webp)."
    )



def build_index_from_paths(
    relative_paths: list[str],
    *,
    index_name: str = "default",
    chunk_size: int = 800,
    overlap: int = 120,
) -> dict[str, Any]:
    base = _files_base()
    chunks_out: list[dict[str, Any]] = []
    files_indexed: list[str] = []
    errors: list[str] = []

    for rel in relative_paths:
        rel = str(rel).strip().lstrip("/\\")
        if not rel:
            continue
        try:
            path = _resolve_scoped_path(base, rel)
        except Exception as e:
            errors.append(f"{rel}: {e}")
            continue
        if not path.exists() or not path.is_file():
            errors.append(f"{rel}: not found")
            continue
        try:
            if is_image_path(path):
                # Keep the binary path for vision; text is only a label for retrieval.
                label = f"image file {rel}"
                chunks_out.append(
                    {
                        "id": hashlib.sha1(f"{rel}:image".encode()).hexdigest()[:12],
                        "source": rel,
                        "chunk_index": 0,
                        "text": label,
                        "vector": _vectorize(label + " photo picture image screenshot"),
                        "kind": "image",
                        "image_path": rel,
                    }
                )
                files_indexed.append(rel)
            else:
                text = extract_text(path)
                parts = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                for i, part in enumerate(parts):
                    chunks_out.append(
                        {
                            "id": hashlib.sha1(f"{rel}:{i}:{part[:40]}".encode()).hexdigest()[:12],
                            "source": rel,
                            "chunk_index": i,
                            "text": part,
                            "vector": _vectorize(part),
                            "kind": "text",
                        }
                    )
                files_indexed.append(rel)
        except Exception as e:
            errors.append(f"{rel}: {e}")

    idx_dir = _index_dir(index_name)
    meta = {
        "index_name": index_name,
        "files": files_indexed,
        "chunk_count": len(chunks_out),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "errors": errors,
    }
    (idx_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Store without huge duplication — vectors included for retrieval
    (idx_dir / "chunks.json").write_text(json.dumps(chunks_out), encoding="utf-8")
    return meta


def query_index(
    query: str,
    *,
    index_name: str = "default",
    top_k: int = 5,
) -> dict[str, Any]:
    idx_dir = _index_dir(index_name)
    chunks_path = idx_dir / "chunks.json"
    if not chunks_path.exists():
        raise ConnectedExecutionError(
            f"RAG index '{index_name}' not found. Run an Index Documents node first, "
            f"or upload files and index them."
        )
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    qv = _vectorize(query)
    scored = []
    for ch in chunks:
        score = _cosine(qv, ch.get("vector") or {})
        scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, min(top_k, 20))]
    results = [
        {
            "score": round(score, 4),
            "source": ch.get("source"),
            "chunk_index": ch.get("chunk_index"),
            "text": ch.get("text"),
        }
        for score, ch in top
        if score > 0
    ]
    # Always return something if index non-empty but scores zero (rare tokens)
    if not results and scored:
        for score, ch in scored[:top_k]:
            results.append(
                {
                    "score": round(score, 4),
                    "source": ch.get("source"),
                    "chunk_index": ch.get("chunk_index"),
                    "text": ch.get("text"),
                }
            )
    context = "\n\n---\n\n".join(
        f"[{r['source']}#{r['chunk_index']} score={r['score']}]\n{r['text']}" for r in results
    )
    # Absolute paths so vision providers can open files regardless of CWD.
    image_rels = []
    for score, ch in scored[: max(top_k * 3, 10)]:
        if ch.get("kind") == "image" or ch.get("image_path"):
            image_rels.append(ch.get("image_path") or ch.get("source"))
    # Also include image hits already in results
    for r in results:
        src = r.get("source")
        # recover kind from original chunk list
        pass
    for ch in chunks:
        if ch.get("kind") == "image":
            rel = ch.get("image_path") or ch.get("source")
            if rel and rel not in image_rels:
                # Prefer images that appear in top text results by source name, else all images in small indexes
                if len([c for c in chunks if c.get("kind") == "image"]) <= 5 or any(
                    r.get("source") == rel for r in results
                ):
                    image_rels.append(rel)

    abs_images = []
    base = _files_base()
    for rel in image_rels:
        try:
            abs_images.append(str(_resolve_scoped_path(base, str(rel)).resolve()))
        except Exception:
            continue

    return {
        "rag_chunks": results,
        "rag_context": context,
        "rag_index": index_name,
        "rag_hit_count": len(results),
        "rag_images": abs_images,
        "image_path": abs_images[0] if abs_images else None,
        # Convenience for Ask AI context budget (prefer this over full rows)
        "table_preview": [{"source": r["source"], "text": r["text"][:400]} for r in results],
    }


def build_rag_index_executor() -> NodeExecutor:
    def execute(node: AgentNode, spec: AgentSpecification, variables: dict[str, Any]) -> dict:
        cfg = node.config or {}
        paths = cfg.get("paths") or cfg.get("files") or []
        if isinstance(paths, str):
            paths = [p.strip() for p in paths.split(",") if p.strip()]
        single = cfg.get("path")
        if single:
            paths = list(paths) + [single]
        # Also allow upstream variable listing uploaded files
        if not paths and isinstance(variables.get("uploaded_files"), list):
            paths = variables["uploaded_files"]
        if not paths:
            raise ConnectedExecutionError(
                f"Node '{node.id}' Index Documents needs config.paths (list of files under agent files dir)."
            )
        index_name = str(cfg.get("index_name") or cfg.get("index") or "default")
        chunk_size = int(cfg.get("chunk_size") or 800)
        overlap = int(cfg.get("overlap") or 120)
        meta = build_index_from_paths(
            [str(p) for p in paths],
            index_name=index_name,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        files = meta.get("files") or []
        errors = list(meta.get("errors") or [])
        base = _files_base()
        abs_images = []
        for rel in files:
            try:
                p = _resolve_scoped_path(base, str(rel))
                if is_image_path(p):
                    abs_images.append(str(p.resolve()))
            except Exception:
                continue
        sample_context = ""
        sample_chunks = []
        try:
            chunks_path = _index_dir(index_name) / "chunks.json"
            if chunks_path.exists():
                all_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
                text_chunks = [c for c in all_chunks if c.get("kind") != "image" and c.get("text")]
                sample_chunks = [
                    {"source": c.get("source"), "chunk_index": c.get("chunk_index"), "text": c.get("text")}
                    for c in text_chunks[:8]
                ]
                bits = []
                for c in sample_chunks:
                    bits.append("[" + str(c.get("source")) + "#" + str(c.get("chunk_index")) + "]\n" + str(c.get("text")))
                sample_context = "\n\n---\n\n".join(bits)
        except Exception:
            sample_context = ""
        if int(meta.get("chunk_count") or 0) == 0 and not abs_images:
            msg = "Indexing produced no text chunks. "
            if errors:
                msg += "Errors: " + str(errors)
            else:
                msg += "For PDFs run: pip install pypdf — and use a text-based PDF (not scan-only)."
            raise ConnectedExecutionError(msg)
        return {
            "rag_index": index_name,
            "rag_files_indexed": files,
            "rag_chunk_count": meta.get("chunk_count"),
            "rag_index_errors": errors,
            "rag_images": abs_images,
            "image_path": abs_images[0] if abs_images else None,
            "rag_context": sample_context,
            "rag_chunks": sample_chunks,
            "rag_hit_count": len(sample_chunks),
        }

    return execute


def build_rag_query_executor() -> NodeExecutor:
    def execute(node: AgentNode, spec: AgentSpecification, variables: dict[str, Any]) -> dict:
        cfg = node.config or {}
        index_name = str(cfg.get("index_name") or cfg.get("index") or "default")
        top_k = int(cfg.get("top_k") or 5)
        q = cfg.get("query") or variables.get("rag_query") or variables.get("question") or ""
        if not str(q).strip():
            # Fall back to last AI / user text if present
            q = variables.get("ai_response") or variables.get("user_message") or ""
        if not str(q).strip():
            raise ConnectedExecutionError(
                f"Node '{node.id}' Query Knowledge needs config.query (or variable question / rag_query)."
            )
        return query_index(str(q), index_name=index_name, top_k=top_k)

    return execute
