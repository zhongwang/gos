"""Load the standalone distilled GenomeOcean GOS observer from Hugging Face."""
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "DOEJGI/GenomeOcean-Sentinel"
# Pinned to the published weights, tokenizer, and custom modeling code.
DEFAULT_MODEL_REVISION = "7ad672818571cb63f7646539d60414e2d271b90b"


class GenomeOceanStudent:
    @classmethod
    def from_checkpoint(cls, model_source, device, checkpoint=None):
        """Load the standalone HF model and validate the shipped normalization.

        The weights, tokenizer, and normalization scalars (``target_mean``,
        ``target_std``, ``max_length``) all come from the published
        ``DOEJGI/GenomeOcean-Sentinel`` model — the local ``.pt`` is no longer
        needed. ``checkpoint`` is accepted for backwards compatibility but is
        only used to cross-check metadata if supplied.
        """
        revision = DEFAULT_MODEL_REVISION if str(model_source) == DEFAULT_MODEL else None
        model = AutoModel.from_pretrained(
            model_source, revision=revision, trust_remote_code=True,
            dtype=torch.float32,
        )
        if model.config.hidden_size != 768 or model.config.architectures != ["GOSStudentForObserver"]:
            raise ValueError("expected the standalone GOS student observer")
        mean = float(model.config.target_mean)
        std = float(model.config.target_std)
        max_length = int(model.config.max_length)
        if not torch.isfinite(torch.tensor([mean, std])).all() or std <= 0:
            raise ValueError("invalid observer normalization")
        if max_length != 200:
            raise ValueError("expected the shipped 200-token student")
        if checkpoint is not None:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
            if (float(saved["target_mean"]), float(saved["target_std"]), int(saved["max_length"])) != (mean, std, max_length):
                raise ValueError("old checkpoint metadata does not match the standalone model")
        model.to(device).eval().requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(model_source, revision=revision)
        if tokenizer.pad_token_id != 3 or tokenizer.pad_token_id != model.config.pad_token_id:
            raise ValueError("tokenizer padding does not match the backbone")
        return model, tokenizer, mean, std, max_length
