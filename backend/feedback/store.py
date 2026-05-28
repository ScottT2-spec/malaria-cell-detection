"""
Correction Store — Persistent storage for doctor feedback.
Uses JSONL for append-only, corruption-resistant storage.
Includes confidence-range matching for finding relevant past corrections.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class CorrectionStore:
    """Append-only JSONL store for doctor corrections with smart retrieval."""

    def __init__(self, corrections_dir: str):
        self.dir = Path(corrections_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.dir / "corrections.jsonl"
        self._cache: list[dict] = []
        self._load()

    def _load(self):
        """Load existing corrections into memory."""
        self._cache = []
        if self.filepath.exists():
            for line in self.filepath.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    self._cache.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    @property
    def count(self) -> int:
        return len(self._cache)

    def save(
        self,
        scan_id: str,
        original: str,
        corrected: str,
        species: Optional[str] = None,
        parasitemia: Optional[str] = None,
        notes: Optional[str] = None,
        doctor_id: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> str:
        """Save a doctor correction. Returns correction ID."""
        correction_id = str(uuid.uuid4())[:12]

        # Build human-readable RAG text
        rag_parts = [
            f"Doctor correction: CNN predicted '{original}' but was corrected to '{corrected}'."
        ]
        if confidence is not None:
            rag_parts.append(f"CNN confidence was {confidence:.1%}.")
        if species:
            rag_parts.append(f"Species identified: {species}.")
        if parasitemia:
            rag_parts.append(f"Parasitemia level: {parasitemia}.")
        if notes:
            rag_parts.append(f"Clinical notes: {notes}")

        record = {
            "id": correction_id,
            "scan_id": scan_id,
            "original_prediction": original,
            "corrected_prediction": corrected,
            "original_confidence": confidence,
            "species": species,
            "parasitemia_level": parasitemia,
            "doctor_notes": notes,
            "doctor_id": doctor_id or "anonymous",
            "rag_text": " ".join(rag_parts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Append to file (atomic-ish: single write + flush)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

        self._cache.append(record)
        return correction_id

    def find_similar(
        self,
        prediction: str,
        confidence: float,
        confidence_range: float = 0.15,
        limit: int = 5,
    ) -> list[dict]:
        """
        Find corrections for similar cases.
        Matches on prediction type AND confidence range (±15% by default).
        More relevant than just matching prediction alone.
        """
        results = []
        for correction in self._cache:
            # Must match prediction type
            if correction["original_prediction"] != prediction:
                continue

            # Score by confidence proximity (closer = more relevant)
            orig_conf = correction.get("original_confidence")
            relevance = 1.0

            if orig_conf is not None:
                conf_diff = abs(orig_conf - confidence)
                if conf_diff > confidence_range:
                    continue  # Too different, skip
                # Higher relevance for closer confidence matches
                relevance = 1.0 - (conf_diff / confidence_range)

            results.append({
                "text": correction["rag_text"],
                "doctor_id": correction.get("doctor_id", "anonymous"),
                "timestamp": correction["timestamp"],
                "relevance": round(relevance, 3),
                "species": correction.get("species"),
            })

        # Sort by relevance (highest first), then recency
        results.sort(key=lambda x: (x["relevance"], x["timestamp"]), reverse=True)
        return results[:limit]

    def stats(self) -> dict:
        """Dashboard statistics for the feedback system."""
        total = len(self._cache)
        if total == 0:
            return {
                "total_corrections": 0,
                "unique_doctors": 0,
                "corrections_by_type": {},
                "species_distribution": {},
                "false_positive_rate": None,
                "false_negative_rate": None,
                "recent": [],
            }

        by_type = {}
        species_dist = {}
        doctors = set()
        false_positives = 0  # CNN said parasitized, doctor said uninfected
        false_negatives = 0  # CNN said uninfected, doctor said parasitized

        for c in self._cache:
            key = f"{c['original_prediction']} → {c['corrected_prediction']}"
            by_type[key] = by_type.get(key, 0) + 1

            sp = c.get("species")
            if sp:
                species_dist[sp] = species_dist.get(sp, 0) + 1

            doctors.add(c.get("doctor_id", "anonymous"))

            if c["original_prediction"] == "parasitized" and c["corrected_prediction"] == "uninfected":
                false_positives += 1
            elif c["original_prediction"] == "uninfected" and c["corrected_prediction"] == "parasitized":
                false_negatives += 1

        recent = self._cache[-5:]
        recent.reverse()

        return {
            "total_corrections": total,
            "unique_doctors": len(doctors),
            "corrections_by_type": by_type,
            "species_distribution": species_dist,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "recent": [
                {
                    "id": r["id"],
                    "original": r["original_prediction"],
                    "corrected": r["corrected_prediction"],
                    "confidence": r.get("original_confidence"),
                    "timestamp": r["timestamp"],
                }
                for r in recent
            ],
        }
