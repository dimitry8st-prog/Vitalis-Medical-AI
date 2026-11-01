from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())


def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
    """Простой bag-of-hashes эмбеддинг без внешних моделей (работает офлайн)."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        vec[idx] += sign
    # bigrams
    for a, b in zip(tokens, tokens[1:]):
        h = int(hashlib.md5(f"{a}_{b}".encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        vec[idx] += 0.5 * sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


@dataclass
class Chunk:
    id: str
    path: str
    title: str
    text: str
    meta: dict


class VaultRAG:
    def __init__(self, vault_dir: Path, index_dir: Path, dim: int = 384):
        self.vault_dir = vault_dir
        self.index_dir = index_dir
        self.dim = dim
        self.chunks: list[Chunk] = []
        self.index: faiss.IndexFlatIP | None = None

    def build(self) -> int:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = []
        vectors: list[np.ndarray] = []

        files = sorted(self.vault_dir.rglob("*.md"))
        for fp in files:
            raw = fp.read_text(encoding="utf-8")
            meta, body = self._split_frontmatter(raw)
            title = meta.get("title") or fp.stem.replace("-", " ")
            parts = self._chunk_text(body, max_len=700)
            for i, part in enumerate(parts):
                cid = f"{fp.relative_to(self.vault_dir).as_posix()}#{i}"
                chunk = Chunk(
                    id=cid,
                    path=str(fp.relative_to(self.vault_dir)),
                    title=title,
                    text=part,
                    meta=meta,
                )
                self.chunks.append(chunk)
                vectors.append(_hash_embed(f"{title}\n{part}", self.dim))

        if not vectors:
            self.index = faiss.IndexFlatIP(self.dim)
            self._persist()
            return 0

        mat = np.vstack(vectors).astype(np.float32)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(mat)
        self._persist()
        return len(self.chunks)

    def load_or_build(self) -> int:
        meta_path = self.index_dir / "chunks.json"
        index_path = self.index_dir / "index.faiss"
        manifest_path = self.index_dir / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}
        if (
            meta_path.exists()
            and index_path.exists()
            and manifest.get("schema_version") == 1
            and manifest.get("vault_fingerprint") == self._vault_fingerprint()
        ):
            self.chunks = [
                Chunk(**c) for c in json.loads(meta_path.read_text(encoding="utf-8"))
            ]
            self.index = faiss.read_index(str(index_path))
            return len(self.chunks)
        return self.build()

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self.index or not self.chunks:
            return []
        q = _hash_embed(query, self.dim).reshape(1, -1)
        scores, idxs = self.index.search(q, min(k, len(self.chunks)))
        out = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            c = self.chunks[i]
            out.append(
                {
                    "id": c.id,
                    "path": c.path,
                    "title": c.title,
                    "text": c.text,
                    "score": float(score),
                    "meta": c.meta,
                }
            )
        return out

    def _persist(self) -> None:
        meta_path = self.index_dir / "chunks.json"
        index_path = self.index_dir / "index.faiss"
        manifest_path = self.index_dir / "manifest.json"
        meta_path.write_text(
            json.dumps([c.__dict__ for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.index is not None:
            faiss.write_index(self.index, str(index_path))
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "vault_fingerprint": self._vault_fingerprint(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _vault_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(self.vault_dir.rglob("*.md")):
            digest.update(path.relative_to(self.vault_dir).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _split_frontmatter(raw: str) -> tuple[dict, str]:
        if not raw.startswith("---"):
            return {}, raw
        parts = raw.split("---", 2)
        if len(parts) < 3:
            return {}, raw
        meta: dict = {}
        for line in parts[1].strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                value = v.strip()
                if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
                    try:
                        value = str(json.loads(value))
                    except (TypeError, ValueError):
                        value = value.strip('"')
                else:
                    value = value.strip("'")
                meta[k.strip()] = value
        return meta, parts[2].strip()

    @staticmethod
    def _chunk_text(text: str, max_len: int = 700) -> list[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paras:
            return [text[:max_len]] if text.strip() else []
        chunks: list[str] = []
        buf = ""
        for p in paras:
            if len(buf) + len(p) + 2 <= max_len:
                buf = f"{buf}\n\n{p}".strip()
            else:
                if buf:
                    chunks.append(buf)
                if len(p) <= max_len:
                    buf = p
                else:
                    for i in range(0, len(p), max_len):
                        chunks.append(p[i : i + max_len])
                    buf = ""
        if buf:
            chunks.append(buf)
        return chunks
