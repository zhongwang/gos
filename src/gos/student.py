"""Distilled GenomeOcean-100M v1.2 observer."""
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class GenomeOceanStudent(nn.Module):
    def __init__(self, snapshot: str | Path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        if self.backbone.config.hidden_size != 768:
            raise ValueError("expected GenomeOcean hidden_size=768")
        self.head = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        )[0]
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1).float()

    @classmethod
    def from_checkpoint(cls, snapshot, checkpoint, device):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("schema_version") != "gos-v4-stage2-distillation-benchmark/1":
            raise ValueError("unsupported student checkpoint schema")
        if saved.get("candidate", {}).get("kind") != "genomeocean":
            raise ValueError("checkpoint must contain the GenomeOcean student")
        model = cls(snapshot)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        model.to(device).eval().requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=True,
        )
        tokenizer.pad_token = "[PAD]"
        if tokenizer.pad_token_id != model.backbone.config.pad_token_id:
            raise ValueError("tokenizer padding does not match the backbone")
        mean, std = float(saved["target_mean"]), float(saved["target_std"])
        if not torch.isfinite(torch.tensor([mean, std])).all() or std <= 0:
            raise ValueError("invalid observer normalization")
        max_length = int(saved["max_length"])
        if max_length != 200:
            raise ValueError("expected the shipped 200-token student")
        return model, tokenizer, mean, std, max_length
