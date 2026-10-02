"""Load the standalone distilled GenomeOcean GOS observer from Hugging Face."""
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "DOEJGI/GenomeOcean-Sentinel"
# Pinned to the published weights, tokenizer, and custom modeling code.
DEFAULT_MODEL_REVISION = "0d91aff1293ad48e71335a6963316a2062c090d5"


class GenomeOceanStudent:
    @classmethod
    def from_checkpoint(cls, model_source, checkpoint, device):
        """Load standalone HF weights and validate the shipped normalization.

        The retained .pt supplies metadata only; its embedded training paths
        are never used. An existing standalone snapshot can be used offline.
        """
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if saved.get("schema_version") != "gos-v4-stage2-distillation-benchmark/1":
            raise ValueError("unsupported student checkpoint schema")
        if saved.get("candidate", {}).get("kind") != "genomeocean":
            raise ValueError("checkpoint must contain the GenomeOcean student")
        mean, std = float(saved["target_mean"]), float(saved["target_std"])
        if not torch.isfinite(torch.tensor([mean, std])).all() or std <= 0:
            raise ValueError("invalid observer normalization")
        max_length = int(saved["max_length"])
        if max_length != 200:
            raise ValueError("expected the shipped 200-token student")
        revision = DEFAULT_MODEL_REVISION if str(model_source) == DEFAULT_MODEL else None
        model = AutoModel.from_pretrained(
            model_source, revision=revision, trust_remote_code=True,
            dtype=torch.float32,
        )
        if model.config.hidden_size != 768 or model.config.architectures != ["GOSStudentForObserver"]:
            raise ValueError("expected the standalone GOS student observer")
        if (model.config.target_mean, model.config.target_std, model.config.max_length) != (mean, std, max_length):
            raise ValueError("standalone model normalization does not match checkpoint metadata")
        model.to(device).eval().requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(model_source, revision=revision)
        if tokenizer.pad_token_id != 3 or tokenizer.pad_token_id != model.config.pad_token_id:
            raise ValueError("tokenizer padding does not match the backbone")
        return model, tokenizer, mean, std, max_length
