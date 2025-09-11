import os
import glob
import re
from typing import List, Dict, Optional, Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-16", errors="ignore") as f:
            return f.read()

def _read_pdf_file(path: str) -> str:
    try:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return ""

def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    for p in paras:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            start = 0
            while start < len(p):
                end = min(start + max_chars, len(p))
                chunks.append(p[start:end])
                if end == len(p):
                    break
                start = max(0, end - overlap)
    return chunks

class RAGIndex:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.local_chunks: List[Dict] = []
        self.external_chunks: List[Dict] = []
        self.use_local = True

    def disable_local_docs(self):
        self.use_local = False
        self._rebuild()

    def reset_external_docs(self):
        self.external_chunks = []
        self._rebuild()

    def add_external_docs(self, docs: Iterable[Dict]):
        # docs: [{text, source}]
        for d in docs:
            if d.get("text") and d.get("source"):
                self.external_chunks.append({"text": d["text"], "source": d["source"]})
        self._rebuild()

    def add_uploaded_files(self, uploaded_files) -> int:
        count = 0
        for uf in uploaded_files:
            name = uf.name
            try:
                if name.lower().endswith(".pdf"):
                    from io import BytesIO
                    bio = BytesIO(uf.read())
                    reader = PdfReader(bio)
                    full = []
                    for p in reader.pages:
                        full.append(p.extract_text() or "")
                    text = "\n".join(full).strip()
                else:
                    text = uf.read().decode("utf-8", errors="ignore")
                chunks = _chunk_text(text)
                for ch in chunks:
                    self.external_chunks.append({"text": ch, "source": f"uploaded:{name}"})
                count += 1
            except Exception:
                continue
        self._rebuild()
        return count

    def build(self):
        # read local files (md, txt, pdf)
        paths = []
        for ext in ("*.md", "*.txt", "*.pdf"):
            paths.extend(glob.glob(os.path.join(self.data_dir, "**", ext), recursive=True))
        local = []
        for p in sorted(set(paths)):
            text = ""
            if p.lower().endswith(".pdf"):
                text = _read_pdf_file(p)
            else:
                text = _read_text_file(p)
            if not text:
                continue
            for ch in _chunk_text(text):
                local.append({"text": ch, "source": p})
        self.local_chunks = local
        self._rebuild()

    def _rebuild(self):
        # merge selected sources
        chunks = []
        if self.use_local:
            chunks.extend(self.local_chunks)
        if self.external_chunks:
            chunks.extend(self.external_chunks)

        if not chunks:
            chunks = [{"text": "No documents found. Please add files into data/ or enable vendor sources.", "source": "N/A"}]

        corpus = [c["text"] for c in chunks]
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
            max_features=60000
        )
        self.matrix = self.vectorizer.fit_transform(corpus)
        self._chunks = chunks  # keep merged view

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        if self.vectorizer is None or self.matrix is None:
            self.build()
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        idxs = sims.argsort()[::-1][:top_k]
        results = []
        for i in idxs:
            results.append({
                "text": self._chunks[i]["text"],
                "source": self._chunks[i]["source"],
                "score": float(sims[i])
            })
        return results
