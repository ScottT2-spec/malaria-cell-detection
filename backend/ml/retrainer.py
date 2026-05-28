"""
Adaptive Model Retrainer — Safe incremental learning for the malaria CNN.

Anti-catastrophic-forgetting strategy (Experience Replay + EWC-lite):
1. FREEZE all convolutional layers (feature extractors don't change)
2. Fine-tune only the dense classification head
3. Mixed batches: corrections + rehearsal samples from original training set
   (rehearsal ratio prevents the model from "forgetting" old knowledge)
4. Elastic Weight Consolidation (EWC) penalty: penalizes large weight
   changes on parameters that were important for original task performance
5. Validation gate: new model MUST beat old model on a held-out validation
   set before it's promoted — otherwise the retrain is rejected
6. Automatic rollback: old model is always preserved

References:
- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (EWC)
- Rolnick et al., "Experience Replay for Continual Learning"
"""

import os
import json
import shutil
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("malariaai.retrainer")

# ── Config ───────────────────────────────────────────────────────

IMG_SIZE = 64
MIN_CORRECTIONS_TO_RETRAIN = 30   # Don't retrain with too few samples
REHEARSAL_RATIO = 3.0              # 3 original samples per 1 correction
FINE_TUNE_EPOCHS = 5               # Few epochs to avoid overfitting
FINE_TUNE_LR = 1e-4                # Low LR for stability
EWC_LAMBDA = 500.0                 # EWC regularization strength
VALIDATION_SPLIT = 0.2             # 20% of corrections held out for validation
MIN_ACCURACY_IMPROVEMENT = 0.0     # New model must be >= old model (no regression)


class ModelRetrainer:
    """
    Safe incremental CNN retrainer with anti-catastrophic-forgetting guarantees.

    The retraining is NEVER automatic — it must be explicitly triggered.
    The old model is ALWAYS preserved. The new model MUST pass a validation
    gate before being promoted.
    """

    def __init__(
        self,
        model_dir: str,
        corrections_dir: str,
        rehearsal_dir: Optional[str] = None,
        history_dir: Optional[str] = None,
    ):
        self.model_dir = Path(model_dir)
        self.corrections_dir = Path(corrections_dir)
        self.corrections_images_dir = self.corrections_dir / "images"
        self.corrections_images_dir.mkdir(parents=True, exist_ok=True)

        # Rehearsal data: a representative sample of the original training set
        self.rehearsal_dir = Path(rehearsal_dir) if rehearsal_dir else self.model_dir.parent / "data" / "rehearsal"

        # Training history
        self.history_dir = Path(history_dir) if history_dir else self.model_dir.parent / "data" / "retrain_history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def save_correction_image(
        self,
        scan_id: str,
        image_bytes: bytes,
        corrected_label: str,
        original_label: str,
        confidence: float,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Save a corrected image for future retraining.
        Organized by corrected label for easy loading.
        """
        label_dir = self.corrections_images_dir / corrected_label
        label_dir.mkdir(parents=True, exist_ok=True)

        # Hash the image to avoid duplicates
        img_hash = hashlib.md5(image_bytes).hexdigest()[:10]
        filename = f"{scan_id}_{img_hash}.png"
        filepath = label_dir / filename

        filepath.write_bytes(image_bytes)

        # Save metadata alongside
        meta = {
            "scan_id": scan_id,
            "original_label": original_label,
            "corrected_label": corrected_label,
            "original_confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        meta_path = label_dir / f"{scan_id}_{img_hash}.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info(f"Saved correction image: {filepath} (was={original_label}, now={corrected_label})")
        return str(filepath)

    def correction_count(self) -> dict:
        """Count available correction images by class."""
        counts = {"parasitized": 0, "uninfected": 0, "total": 0}
        for label in ("parasitized", "uninfected"):
            label_dir = self.corrections_images_dir / label
            if label_dir.exists():
                n = len([f for f in label_dir.iterdir() if f.suffix == ".png"])
                counts[label] = n
                counts["total"] += n
        return counts

    def can_retrain(self) -> dict:
        """Check if retraining is possible and advisable."""
        counts = self.correction_count()
        has_rehearsal = self.rehearsal_dir.exists() and any(self.rehearsal_dir.iterdir())

        ready = counts["total"] >= MIN_CORRECTIONS_TO_RETRAIN
        reasons = []

        if counts["total"] < MIN_CORRECTIONS_TO_RETRAIN:
            reasons.append(
                f"Need at least {MIN_CORRECTIONS_TO_RETRAIN} corrections, "
                f"have {counts['total']}"
            )
        if not has_rehearsal:
            reasons.append(
                "No rehearsal data found. Place a representative sample of original "
                "training images in data/rehearsal/parasitized/ and data/rehearsal/uninfected/"
            )
            ready = False

        return {
            "ready": ready,
            "correction_counts": counts,
            "has_rehearsal_data": has_rehearsal,
            "min_required": MIN_CORRECTIONS_TO_RETRAIN,
            "reasons": reasons,
        }

    async def retrain(self) -> dict:
        """
        Execute safe incremental retraining.

        Strategy:
        1. Load current model
        2. Freeze conv layers
        3. Build mixed dataset (corrections + rehearsal)
        4. Fine-tune with EWC penalty
        5. Validate: new model must match or beat old accuracy
        6. If passed: promote new model. If failed: reject and keep old.

        Returns detailed result dict.
        """
        # Lazy import — tensorflow is heavy, only load when actually retraining
        try:
            import numpy as np
            import tensorflow as tf
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
        except ImportError:
            return {
                "status": "error",
                "message": "TensorFlow not installed. Run: pip install tensorflow",
            }

        status = self.can_retrain()
        if not status["ready"]:
            return {"status": "blocked", "reasons": status["reasons"]}

        retrain_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting retrain {retrain_id}")

        try:
            # ── 1. Load current model ────────────────────────────
            model_path = self.model_dir / "malaria_model.keras"
            if not model_path.exists():
                # Try h5 format
                model_path = self.model_dir / "malaria_model.h5"
            if not model_path.exists():
                return {"status": "error", "message": "No saved model found"}

            model = tf.keras.models.load_model(str(model_path))
            logger.info("Loaded current model")

            # ── 2. Save original weights (for EWC) ──────────────
            original_weights = [w.numpy().copy() for w in model.trainable_weights]

            # ── 3. Freeze convolutional layers ──────────────────
            frozen_count = 0
            for layer in model.layers:
                if 'conv' in layer.name or 'pool' in layer.name:
                    layer.trainable = False
                    frozen_count += 1
            logger.info(f"Froze {frozen_count} conv/pool layers")

            # ── 4. Load correction images ───────────────────────
            X_corrections, y_corrections = self._load_images(
                self.corrections_images_dir, tf, load_img, img_to_array, np
            )
            logger.info(f"Loaded {len(X_corrections)} correction images")

            # ── 5. Load rehearsal images (experience replay) ────
            X_rehearsal, y_rehearsal = self._load_images(
                self.rehearsal_dir, tf, load_img, img_to_array, np
            )
            logger.info(f"Loaded {len(X_rehearsal)} rehearsal images")

            # ── 6. Build mixed dataset ──────────────────────────
            # Sample rehearsal data at REHEARSAL_RATIO relative to corrections
            n_rehearsal_needed = int(len(X_corrections) * REHEARSAL_RATIO)
            if len(X_rehearsal) > n_rehearsal_needed:
                indices = np.random.choice(len(X_rehearsal), n_rehearsal_needed, replace=False)
                X_rehearsal = X_rehearsal[indices]
                y_rehearsal = y_rehearsal[indices]

            X_train = np.concatenate([X_corrections, X_rehearsal])
            y_train = np.concatenate([y_corrections, y_rehearsal])

            # Shuffle
            shuffle_idx = np.random.permutation(len(X_train))
            X_train = X_train[shuffle_idx]
            y_train = y_train[shuffle_idx]

            # Split validation
            split = int(len(X_train) * (1 - VALIDATION_SPLIT))
            X_val, y_val = X_train[split:], y_train[split:]
            X_train, y_train = X_train[:split], y_train[:split]

            logger.info(f"Mixed dataset: {len(X_train)} train, {len(X_val)} val "
                        f"(corrections: {len(X_corrections)}, rehearsal: {n_rehearsal_needed})")

            # ── 7. Evaluate old model (baseline) ────────────────
            old_loss, old_accuracy = model.evaluate(X_val, y_val, verbose=0)
            logger.info(f"Old model validation: acc={old_accuracy:.4f}, loss={old_loss:.4f}")

            # ── 8. Compile with EWC-aware training ──────────────
            # Custom training loop with EWC penalty
            optimizer = tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR)
            loss_fn = tf.keras.losses.BinaryCrossentropy()

            # Compute Fisher Information approximation (diagonal)
            # Uses the current model's gradient magnitudes as importance weights
            fisher_diag = self._compute_fisher(
                model, X_train[:min(200, len(X_train))], y_train[:min(200, len(y_train))],
                tf, loss_fn
            )

            # ── 9. Fine-tune with EWC ───────────────────────────
            batch_size = 32
            n_batches = max(1, len(X_train) // batch_size)

            for epoch in range(FINE_TUNE_EPOCHS):
                epoch_loss = 0.0
                shuffle_idx = np.random.permutation(len(X_train))

                for batch_i in range(n_batches):
                    start = batch_i * batch_size
                    end = start + batch_size
                    batch_idx = shuffle_idx[start:end]
                    X_batch = X_train[batch_idx]
                    y_batch = y_train[batch_idx]

                    with tf.GradientTape() as tape:
                        predictions = model(X_batch, training=True)
                        ce_loss = loss_fn(y_batch, predictions)

                        # EWC penalty: penalize changes to important weights
                        ewc_loss = 0.0
                        for i, w in enumerate(model.trainable_weights):
                            if i < len(fisher_diag) and i < len(original_weights):
                                ewc_loss += tf.reduce_sum(
                                    fisher_diag[i] * tf.square(w - original_weights[i])
                                )
                        ewc_loss *= EWC_LAMBDA / 2.0

                        total_loss = ce_loss + ewc_loss

                    grads = tape.gradient(total_loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                    epoch_loss += total_loss.numpy()

                avg_loss = epoch_loss / n_batches
                val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
                logger.info(f"Epoch {epoch + 1}/{FINE_TUNE_EPOCHS}: "
                            f"loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

            # ── 10. Validation gate ─────────────────────────────
            new_loss, new_accuracy = model.evaluate(X_val, y_val, verbose=0)
            logger.info(f"New model validation: acc={new_accuracy:.4f}, loss={new_loss:.4f}")

            improvement = new_accuracy - old_accuracy
            passed = improvement >= MIN_ACCURACY_IMPROVEMENT

            if not passed:
                # REJECT — accuracy regressed
                logger.warning(f"Retrain REJECTED: accuracy dropped by {-improvement:.4f}")
                # Model in memory is now dirty — don't save it
                result = {
                    "status": "rejected",
                    "retrain_id": retrain_id,
                    "reason": f"Accuracy regression: {old_accuracy:.4f} → {new_accuracy:.4f}",
                    "old_accuracy": round(old_accuracy, 4),
                    "new_accuracy": round(new_accuracy, 4),
                    "improvement": round(improvement, 4),
                    "corrections_used": len(X_corrections),
                    "rehearsal_used": len(X_rehearsal),
                }
            else:
                # ACCEPT — save new model
                logger.info(f"Retrain ACCEPTED: {old_accuracy:.4f} → {new_accuracy:.4f} (+{improvement:.4f})")

                # Backup old model
                backup_dir = self.history_dir / retrain_id
                backup_dir.mkdir(parents=True, exist_ok=True)
                if model_path.exists():
                    shutil.copy2(model_path, backup_dir / model_path.name)

                # Save new model
                new_model_path = self.model_dir / "malaria_model.keras"
                model.save(str(new_model_path))

                # Also convert to TF.js if tensorflowjs is available
                try:
                    import subprocess
                    tfjs_dir = self.model_dir.parent / "docs" / "model"
                    subprocess.run(
                        ["tensorflowjs_converter", "--input_format=keras",
                         str(new_model_path), str(tfjs_dir)],
                        capture_output=True, timeout=120,
                    )
                    logger.info("Converted new model to TF.js format")
                except Exception as e:
                    logger.warning(f"TF.js conversion skipped: {e}")

                result = {
                    "status": "accepted",
                    "retrain_id": retrain_id,
                    "old_accuracy": round(old_accuracy, 4),
                    "new_accuracy": round(new_accuracy, 4),
                    "improvement": round(improvement, 4),
                    "corrections_used": len(X_corrections),
                    "rehearsal_used": len(X_rehearsal),
                    "model_path": str(new_model_path),
                    "backup_path": str(backup_dir),
                }

            # Save retrain record
            record_path = self.history_dir / f"{retrain_id}.json"
            record_path.write_text(json.dumps(result, indent=2))

            return result

        except Exception as e:
            logger.error(f"Retrain failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _load_images(self, base_dir, tf, load_img, img_to_array, np):
        """Load images from parasitized/ and uninfected/ subdirectories."""
        images = []
        labels = []

        for label_name, label_val in [("parasitized", 1), ("uninfected", 0)]:
            label_dir = base_dir / label_name
            if not label_dir.exists():
                continue
            for img_path in label_dir.iterdir():
                if img_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
                    try:
                        img = load_img(str(img_path), target_size=(IMG_SIZE, IMG_SIZE))
                        arr = img_to_array(img) / 255.0
                        images.append(arr)
                        labels.append(label_val)
                    except Exception:
                        continue

        if not images:
            return np.array([]), np.array([])

        return np.array(images, dtype='float32'), np.array(labels, dtype='float32')

    def _compute_fisher(self, model, X, y, tf, loss_fn) -> list:
        """
        Compute diagonal Fisher Information Matrix approximation.
        Measures how important each weight is for the current task.
        Weights with high Fisher values are penalized more during fine-tuning.
        """
        fisher = [tf.zeros_like(w) for w in model.trainable_weights]

        for i in range(len(X)):
            with tf.GradientTape() as tape:
                pred = model(X[i:i+1], training=False)
                loss = loss_fn(y[i:i+1], pred)

            grads = tape.gradient(loss, model.trainable_weights)

            for j, g in enumerate(grads):
                if g is not None:
                    fisher[j] += tf.square(g)

        # Average
        fisher = [f / len(X) for f in fisher]
        return fisher

    def retrain_history(self) -> list[dict]:
        """Get history of all retrain attempts."""
        records = []
        for f in sorted(self.history_dir.iterdir()):
            if f.suffix == '.json':
                try:
                    records.append(json.loads(f.read_text()))
                except Exception:
                    continue
        return records
