"""Retrieval over the business glossary (knowledge/glossary.md).

Each "## " section of the glossary is one entry. They are embedded with
Chroma's built-in all-MiniLM-L6-v2 model, which runs locally on CPU via
ONNX: no API key, no per-call cost, and small enough to deploy on a free host.

The store is rebuilt automatically whenever the glossary file changes, so
editing a definition never leaves stale embeddings behind.
"""
import hashlib
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent.parent
GLOSSARY = ROOT / "knowledge" / "glossary.md"
STORE = ROOT / "knowledge_store"


def _entries(text: str) -> list[str]:
    sections = text.split("\n## ")[1:]          # drop the file header
    return ["## " + s.strip() for s in sections]


class Glossary:
    def __init__(self, path: Path = GLOSSARY):
        text = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        client = chromadb.PersistentClient(path=str(STORE),
                                           settings=chromadb.Settings(anonymized_telemetry=False))
        name = f"glossary_{digest}"
        entries = _entries(text)
        existing = {c.name for c in client.list_collections()}
        if name in existing and client.get_collection(name).count() == len(entries):
            self.collection = client.get_collection(name)
            return
        # Glossary changed, or an earlier build was interrupted: rebuild from scratch.
        for old in existing:
            if old.startswith("glossary_"):
                client.delete_collection(old)
        self.collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        self.collection.add(ids=[f"e{i}" for i in range(len(entries))], documents=entries)

    def search(self, question: str, k: int = 4) -> list[str]:
        res = self.collection.query(query_texts=[question], n_results=k)
        return res["documents"][0]
