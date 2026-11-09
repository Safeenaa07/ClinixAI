"""Module 6: evidence retrieval + verification over a curated knowledge base.
Entries are loaded from knowledge/*.md. Retrieval is TF-IDF (no network, no API key).
Verification marks a finding as supported only if the best-matching entry is about that
finding. A finding with no entry is flagged "no supporting reference"."""
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

KEYS = ("Source", "What", "Matters", "Next")


def norm(label: str) -> str:
    return label.replace("_", " ").strip().lower()


def _parse(text: str, fname: str):
    entries, cur = [], None
    for line in text.splitlines():
        if line.startswith("## "):
            if cur:
                entries.append(cur)
            cur = {"title": line[3:].strip(), "source": "", "what": "", "matters": "", "next": "", "file": fname}
        elif cur is not None:
            for k in KEYS:
                if line.startswith(k + ":"):
                    cur[k.lower()] = line[len(k) + 1:].strip()
    if cur:
        entries.append(cur)
    return entries


class KnowledgeBase:
    def __init__(self, folder: Path):
        self.entries = []
        for f in sorted(folder.glob("*.md")):
            self.entries += _parse(f.read_text(encoding="utf-8"), f.name)
        self.by_title = {norm(e["title"]): e for e in self.entries}
        self._docs = [f'{e["title"]}. {e["what"]} {e["matters"]} {e["next"]}' for e in self.entries]
        if self.entries:
            self._vec = TfidfVectorizer(stop_words="english").fit(self._docs)
            self._mat = self._vec.transform(self._docs)

    def lookup(self, label: str):
        return self.by_title.get(norm(label))

    def retrieve(self, query: str, k: int = 2):
        if not self.entries:
            return []
        sims = linear_kernel(self._vec.transform([query]), self._mat)[0]
        order = sims.argsort()[::-1][:k]
        return [dict(self.entries[i], score=round(float(sims[i]), 3)) for i in order]

    def verify(self, label: str) -> dict:
        """Supported = the knowledge base has an entry for this finding and that entry is
        retrievable by the finding's name. No entry -> flagged as unsupported."""
        e = self.lookup(label)
        if not e:
            return {"supported": False, "score": 0.0, "title": None, "source": None}
        hit = next((h for h in self.retrieve(norm(label), len(self.entries)) if h["title"] == e["title"]), None)
        score = hit["score"] if hit else 0.0
        return {"supported": score >= 0.05, "score": score, "title": e["title"], "source": e["source"]}
