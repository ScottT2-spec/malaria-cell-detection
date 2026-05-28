"""
RAG Engine — Retrieval-Augmented Generation for MalariaAI

Two knowledge sources:
1. Static guidelines (WHO, CDC, etc.) — loaded from data/guidelines/
2. Dynamic corrections (doctor feedback) — loaded from data/corrections/

Uses BM25 (Okapi) ranking — the industry standard for text retrieval.
Lightweight, zero external dependencies, runs anywhere.
"""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter


# ── Medical domain stopwords (removed from queries for better precision) ──
MEDICAL_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "ought",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "just", "about",
    "above", "after", "again", "also", "because", "before", "below",
    "between", "during", "for", "from", "further", "here", "how",
    "into", "its", "itself", "of", "on", "only", "our", "out", "own",
    "same", "she", "that", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "under", "until", "up",
    "what", "when", "where", "which", "while", "who", "whom", "why",
    "with", "you", "your",
})

# ── Medical synonym expansion (query-time) ──
MEDICAL_SYNONYMS = {
    "parasitized": ["infected", "positive", "malaria", "parasite", "plasmodium"],
    "uninfected": ["negative", "healthy", "normal", "clean", "unparasitized"],
    "treatment": ["therapy", "drug", "medication", "dosage", "regimen", "act"],
    "severe": ["critical", "emergency", "danger", "complicated", "high"],
    "falciparum": ["pf", "plasmodium falciparum"],
    "vivax": ["pv", "plasmodium vivax"],
    "diagnosis": ["detection", "screening", "microscopy", "test", "rdt"],
    "pregnant": ["pregnancy", "maternal", "antenatal", "trimester"],
    "child": ["children", "pediatric", "infant", "under five", "paediatric"],
    "resistance": ["resistant", "drug resistance", "artemisinin resistance"],
}


class RAGEngine:
    """BM25-powered RAG engine with query expansion and medical domain awareness."""

    # BM25 hyperparameters (tuned for medical documents)
    BM25_K1 = 1.5   # Term frequency saturation
    BM25_B = 0.75   # Length normalization

    def __init__(self, guidelines_dir: str, corrections_dir: str, embeddings_dir: str):
        self.guidelines_dir = Path(guidelines_dir)
        self.corrections_dir = Path(corrections_dir)
        self.embeddings_dir = Path(embeddings_dir)

        self.chunks: list[dict] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: list[Counter] = []
        self.idf: dict[str, float] = {}
        self.avg_dl: float = 0.0

        self._guideline_count = 0
        self._correction_count = 0
        self._last_updated = None

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def guideline_count(self) -> int:
        return self._guideline_count

    @property
    def correction_count(self) -> int:
        return self._correction_count

    @property
    def last_updated(self) -> str:
        return self._last_updated or "never"

    # ── Loading ──────────────────────────────────────────────────

    def load(self):
        """Load all documents and build the BM25 index."""
        self.chunks = []
        self._load_guidelines()
        self._load_corrections()
        self._build_index()
        self._last_updated = datetime.now(timezone.utc).isoformat()
        print(f"📚 RAG loaded: {len(self.chunks)} chunks "
              f"({self._guideline_count} guidelines, {self._correction_count} corrections)")

    def _load_guidelines(self):
        """Load and chunk guideline documents with section-aware splitting."""
        if not self.guidelines_dir.exists():
            self.guidelines_dir.mkdir(parents=True, exist_ok=True)
            return

        count = 0
        for filepath in sorted(self.guidelines_dir.iterdir()):
            if filepath.suffix in ('.txt', '.md', '.json'):
                count += 1
                text = filepath.read_text(encoding='utf-8', errors='ignore')
                source_name = filepath.stem.replace('_', ' ').title()

                if filepath.suffix == '.json':
                    try:
                        data = json.loads(text)
                        if isinstance(data, list):
                            for i, item in enumerate(data):
                                chunk_text = item.get('text', str(item))
                                self.chunks.append({
                                    "id": f"guideline-{filepath.stem}-{i}",
                                    "text": chunk_text,
                                    "source": item.get('source', source_name),
                                    "type": "guideline",
                                    "metadata": item.get('metadata', {}),
                                })
                        continue
                    except json.JSONDecodeError:
                        pass

                # Section-aware chunking with header preservation
                chunks = self._chunk_text(text, max_chunk=500, overlap=50)
                for i, chunk in enumerate(chunks):
                    self.chunks.append({
                        "id": f"guideline-{filepath.stem}-{i}",
                        "text": chunk,
                        "source": source_name,
                        "type": "guideline",
                        "metadata": {"file": filepath.name, "chunk_index": i},
                    })

        self._guideline_count = count

    def _load_corrections(self):
        """Load doctor corrections as RAG entries."""
        if not self.corrections_dir.exists():
            self.corrections_dir.mkdir(parents=True, exist_ok=True)
            return

        count = 0
        corrections_file = self.corrections_dir / "corrections.jsonl"
        if corrections_file.exists():
            for line in corrections_file.read_text(encoding='utf-8').strip().split('\n'):
                if not line.strip():
                    continue
                try:
                    correction = json.loads(line)
                    count += 1
                    self.chunks.append({
                        "id": f"correction-{correction.get('id', count)}",
                        "text": correction.get("rag_text", ""),
                        "source": f"Doctor Correction #{count}",
                        "type": "correction",
                        "metadata": correction,
                    })
                except json.JSONDecodeError:
                    continue

        self._correction_count = count

    # ── Chunking ─────────────────────────────────────────────────

    def _chunk_text(self, text: str, max_chunk: int = 500, overlap: int = 50) -> list[str]:
        """
        Section-aware chunking that preserves markdown headers as context.
        Each chunk retains its parent section header for better retrieval.
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = ""
        current_header = ""

        for line in lines:
            stripped = line.strip()

            # Track section headers
            if re.match(r'^#{1,3}\s', stripped):
                # Flush current chunk if adding header would exceed limit
                if current_chunk and len(current_chunk) + len(stripped) > max_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with header context
                    current_chunk = current_header + "\n" if current_header else ""
                current_header = stripped

            # Accumulate content
            if len(current_chunk) + len(line) + 1 > max_chunk:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # Overlap: keep last N words for continuity
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else ""
                current_chunk = current_header + "\n" + overlap_text + "\n" + line if current_header else line
            else:
                current_chunk += ("\n" + line) if current_chunk else line

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Filter out empty/tiny chunks
        return [c for c in chunks if len(c.split()) >= 5]

    # ── Tokenization ─────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Medical-aware tokenization with stopword removal."""
        text = text.lower()
        # Preserve hyphenated medical terms (e.g., "artemisinin-based")
        text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1 and t not in MEDICAL_STOPWORDS]

    def _expand_query(self, tokens: list[str]) -> list[str]:
        """
        Query expansion with medical synonyms.
        Adds related terms so "parasitized" also matches "infected", "malaria", etc.
        """
        expanded = list(tokens)
        for token in tokens:
            if token in MEDICAL_SYNONYMS:
                for synonym in MEDICAL_SYNONYMS[token]:
                    syn_tokens = synonym.lower().split()
                    expanded.extend(syn_tokens)
        # Deduplicate while preserving order
        seen = set()
        result = []
        for t in expanded:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    # ── BM25 Index ───────────────────────────────────────────────

    def _build_index(self):
        """Build BM25 index over all chunks."""
        if not self.chunks:
            return

        n_docs = len(self.chunks)

        # Tokenize all documents
        self.doc_tokens = []
        self.doc_freqs = []
        total_length = 0
        df = Counter()

        for chunk in self.chunks:
            tokens = self._tokenize(chunk["text"])
            self.doc_tokens.append(tokens)
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            total_length += len(tokens)

            for token in set(tokens):
                df[token] += 1

        # Average document length
        self.avg_dl = total_length / n_docs if n_docs > 0 else 1.0

        # IDF (with smoothing to avoid negative values)
        self.idf = {}
        for token, freq in df.items():
            # Standard BM25 IDF formula
            self.idf[token] = math.log(
                (n_docs - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def _bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document against query tokens."""
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = len(self.doc_tokens[doc_idx])

        if doc_len == 0:
            return 0.0

        score = 0.0
        for token in query_tokens:
            if token not in doc_freq:
                continue

            tf = doc_freq[token]
            idf = self.idf.get(token, 0.0)

            # BM25 term frequency component
            tf_component = (tf * (self.BM25_K1 + 1)) / (
                tf + self.BM25_K1 * (1 - self.BM25_B + self.BM25_B * doc_len / self.avg_dl)
            )

            score += idf * tf_component

        return score

    # ── Retrieval ────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        boost_corrections: bool = True,
    ) -> list[dict]:
        """
        Retrieve top-k most relevant chunks using BM25 with query expansion.

        Args:
            query: Search query
            top_k: Maximum results to return
            min_score: Minimum BM25 score threshold
            boost_corrections: Give 1.2x boost to doctor corrections (recent clinical experience)
        """
        if not self.chunks or not self.doc_tokens:
            return []

        # Tokenize and expand query
        raw_tokens = self._tokenize(query)
        query_tokens = self._expand_query(raw_tokens)

        # Score all chunks
        scores = []
        for i in range(len(self.chunks)):
            score = self._bm25_score(query_tokens, i)

            # Boost doctor corrections (real-world clinical experience is valuable)
            if boost_corrections and self.chunks[i]["type"] == "correction":
                score *= 1.2

            if score > min_score:
                scores.append((score, i))

        # Sort by score descending
        scores.sort(reverse=True)

        # Deduplicate: skip chunks with >80% token overlap with already-selected chunks
        results = []
        selected_token_sets = []

        for score, idx in scores:
            if len(results) >= top_k:
                break

            chunk_tokens = set(self.doc_tokens[idx])

            # Check overlap with already selected
            is_duplicate = False
            for prev_tokens in selected_token_sets:
                if not chunk_tokens or not prev_tokens:
                    continue
                overlap = len(chunk_tokens & prev_tokens) / min(len(chunk_tokens), len(prev_tokens))
                if overlap > 0.8:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            selected_token_sets.append(chunk_tokens)
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "type": chunk["type"],
                "score": round(score, 4),
                "chunk_id": chunk["id"],
            })

        return results

    def retrieve_for_analysis(self, prediction: str, confidence: float) -> list[dict]:
        """
        Specialized retrieval for clinical analysis.
        Builds a smart query based on the CNN results to pull the most relevant guidelines.
        """
        queries = []

        if prediction == "parasitized":
            queries.append("malaria treatment parasitized infected positive")
            if confidence < 0.7:
                queries.append("low confidence diagnosis verification microscopy quality")
            if confidence > 0.95:
                queries.append("severe malaria high parasitemia emergency treatment")
            queries.append("WHO ACT artemisinin treatment dosage")
        else:
            queries.append("uninfected negative malaria screening")
            if confidence < 0.7:
                queries.append("false negative retest clinical symptoms persistent fever")
            queries.append("malaria prevention follow up")

        # Merge results from all queries, deduplicated
        all_results = {}
        for query in queries:
            for result in self.retrieve(query, top_k=3):
                cid = result["chunk_id"]
                if cid not in all_results or result["score"] > all_results[cid]["score"]:
                    all_results[cid] = result

        # Sort by score and return top results
        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:6]

    # ── Live Updates ─────────────────────────────────────────────

    def add_correction(self, correction_id: str, text: str) -> bool:
        """Add a new correction to the RAG index in real-time (no restart needed)."""
        try:
            new_chunk = {
                "id": f"correction-{correction_id}",
                "text": text,
                "source": f"Doctor Correction #{correction_id}",
                "type": "correction",
                "metadata": {"correction_id": correction_id},
            }
            self.chunks.append(new_chunk)
            self._correction_count += 1

            # Incrementally update index (rebuild is fast for small corpus)
            self._build_index()

            self._last_updated = datetime.now(timezone.utc).isoformat()
            return True
        except Exception as e:
            print(f"⚠️ Failed to add correction to RAG: {e}")
            return False
