"""CPU features plus the distilled observer, using the production decision rule."""
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .decision import DecisionHead, sigmoid
from .features import extract_cpu_features
from .student import DEFAULT_MODEL, GenomeOceanStudent


class GOSDetector:
    def __init__(self, *, model=DEFAULT_MODEL, decision_head=None, device="cpu", dtype="float32", checkpoint=None):
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")
        if dtype == "bfloat16" and self.device.type != "cuda":
            raise ValueError("bfloat16 inference requires CUDA; use float32 on CPU")
        self.dtype = dtype
        torch.set_float32_matmul_precision("high")
        if decision_head is None:
            decision_head = Path(__file__).resolve().parents[2] / "models/gos_detector/decision_head.json"
            if not decision_head.is_file():
                raise ValueError(
                    "decision_head.json not found next to the package; pass --decision-head explicitly"
                )
        self.decision = DecisionHead(decision_head)
        (self.student, self.tokenizer, self.target_mean, self.target_std,
         self.max_length) = GenomeOceanStudent.from_checkpoint(model, self.device, checkpoint=checkpoint)

    def scan_batch(self, sequences: list[str]) -> list[dict]:
        if not sequences:
            return []
        cpu_features = np.vstack([extract_cpu_features(sequence) for sequence in sequences])
        encoded = self.tokenizer(
            sequences, return_tensors="pt", padding="max_length", truncation=True,
            max_length=self.max_length, return_token_type_ids=False,
        )
        batch = {key: value.to(self.device) for key, value in encoded.items()
                 if key in {"input_ids", "attention_mask"}}
        precision = (torch.autocast("cuda", dtype=torch.bfloat16)
                     if self.dtype == "bfloat16" else nullcontext())
        with torch.inference_mode(), precision:
            pred_norm = self.student(**batch)
        observer_logits = (pred_norm.float() * self.target_std + self.target_mean).cpu().numpy().astype(np.float64)
        base_logits = self.decision.base_logits(cpu_features)
        total_logits = base_logits + observer_logits
        if not np.isfinite(total_logits).all():
            raise ValueError("detector produced non-finite logits")
        records = []
        for sequence, base, observer, total in zip(sequences, base_logits, observer_logits, total_logits):
            is_ai = bool(total >= self.decision.threshold_logit)
            records.append({
                "length": len(sequence),
                "prediction": int(is_ai),
                "call": "AI" if is_ai else "Natural",
                "router_confidence": sigmoid(float(total)),
                "reason": "total_logit_at_or_above_threshold" if is_ai else "total_logit_below_threshold",
                "base_logit": float(base),
                "observer_logit": float(observer),
                "total_logit": float(total),
            })
        return records

    def scan(self, sequence: str) -> dict:
        return self.scan_batch([sequence])[0]
