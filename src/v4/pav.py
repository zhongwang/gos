"""Signed positional-attention-variance (pAV) evidence for GOS v4.

This module intentionally has no module-level PyTorch or Transformers import.
The NT observer is constructed only when a pAV stage without an injected
observer is actually asked to score a window.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np

from .artifacts import ArtifactCompatibilityError, artifact_is_certified, validate_artifact
from .preprocessing import PreparedSequence, ScoringWindow, pav_candidate_windows
from .types import StageEvidence, StageStatus, WindowEvidence


_PAV_ARTIFACT_KIND = "pav"
_PAV_CACHE_SCHEMA = "gos-v4-pav-cache/1"
_PAV_CACHE_JSONL_SCHEMA = "gos-v4-pav-cache-jsonl/2"
_LEGACY_PAV_CACHE_JSONL_SCHEMA = "gos-v4-pav-cache-jsonl/1"
_DEFAULT_MODEL_NAME = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
_DEFAULT_REVISION = "main"
_DEFAULT_MAX_LENGTH = 200
_SUPPORTED_NORMALIZATIONS = frozenset({"none", "token_count_squared"})
_OBSERVER_CONFIG_FIELDS = (
    "model_name",
    "revision",
    "tokenizer_identity",
    "max_length",
    "normalization",
)
_OBSERVER_CONFIG_KEYS = frozenset(_OBSERVER_CONFIG_FIELDS)
_REQUIRED_PAYLOAD_KEYS = frozenset(
    {
        "estimator",
        "scaler",
        "clip_lower",
        "clip_upper",
        "selected_heads",
        "signs",
        "centers",
        "scales",
        "window_threshold",
    }
)
_LEGACY_PAYLOAD_KEYS = frozenset({"model", "heads", "clip_low", "clip_high", "threshold", "selection"})


@dataclass(frozen=True)
class ObserverFeatures:
    """Raw observer vectors cached before pAV head selection."""

    pav: np.ndarray
    layer24_mean: np.ndarray | None = None
    attention_entropy: np.ndarray | None = None


@dataclass(frozen=True)
class SignedHeadSelection:
    """A direction-preserving pAV head selection and natural reference scale."""

    heads: tuple[int, ...]
    signs: tuple[float, ...]
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    aucs: tuple[float, ...]


@dataclass(frozen=True)
class PAVAggregates:
    """Window-level pAV availability and score summaries.

    ``mean``/``maximum`` are optional classifier evidence scores.  They are
    deliberately not a provenance prediction: low pAV evidence never becomes a
    numeric natural call in this stage.
    """

    available: bool
    n_windows: int
    n_scored: int
    scored_coverage: float
    mean: float | None
    maximum: float | None
    transformed_mean: tuple[float, ...] | None
    layer24_mean: tuple[float, ...] | None = None
    attention_entropy: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ObserverConfig(Mapping[str, Any]):
    """Every observer setting that can change persisted pAV values."""

    model_name: str
    revision: str
    tokenizer_identity: str
    max_length: int
    normalization: str

    def __post_init__(self) -> None:
        for field in ("model_name", "revision", "tokenizer_identity"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"observer_config.{field} must be a non-empty string")
        if type(self.max_length) is not int or self.max_length < 1:
            raise ValueError("observer_config.max_length must be a positive integer")
        if self.normalization not in _SUPPORTED_NORMALIZATIONS:
            raise ValueError("observer_config.normalization is unsupported")

    def __getitem__(self, field: str) -> Any:
        if field not in _OBSERVER_CONFIG_KEYS:
            raise KeyError(field)
        return getattr(self, field)

    def __iter__(self):
        return iter(_OBSERVER_CONFIG_FIELDS)

    def __len__(self) -> int:
        return len(_OBSERVER_CONFIG_FIELDS)

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-serializable artifact representation."""

        return {field: getattr(self, field) for field in _OBSERVER_CONFIG_FIELDS}

    @property
    def fingerprint(self) -> str:
        """Return SHA-256 over the canonical JSON artifact representation."""

        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ObserverConfig":
        if not isinstance(value, Mapping) or set(value) != _OBSERVER_CONFIG_KEYS:
            raise ValueError(
                "observer_config must contain model_name, revision, "
                "tokenizer_identity, max_length, and normalization"
            )
        return cls(**dict(value))


class ConfiguredObserver:
    """Typed adapter for injecting a lightweight observer with exact provenance."""

    def __init__(
        self,
        observer: Callable[[str], Any],
        observer_config: ObserverConfig,
        *,
        observer_id: str | None = None,
    ) -> None:
        if not callable(observer):
            raise TypeError("observer must be callable")
        if not isinstance(observer_config, ObserverConfig):
            raise TypeError("observer_config must be an ObserverConfig")
        if observer_id is not None and (
            not isinstance(observer_id, str) or not observer_id
        ):
            raise ValueError("observer_id must be a non-empty string")
        effective_id = observer_config.model_name if observer_id is None else observer_id
        self._observer = observer
        self.observer_config = observer_config
        self.observer_fingerprint = observer_config.fingerprint
        self.observer_id = effective_id

    def __call__(self, sequence: str) -> Any:
        return self._observer(sequence)


def _finite_vector(value: Any, *, field: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a numeric vector") from error
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{field} must be a non-empty finite one-dimensional vector")
    return vector.copy()


def observer_features(value: Any) -> ObserverFeatures:
    """Coerce legacy pAV vectors or rich observer mappings into typed features."""

    if isinstance(value, ObserverFeatures):
        return ObserverFeatures(
            pav=_finite_vector(value.pav, field="observer pav"),
            layer24_mean=(
                _finite_vector(value.layer24_mean, field="observer layer24_mean")
                if value.layer24_mean is not None
                else None
            ),
            attention_entropy=(
                _finite_vector(
                    value.attention_entropy,
                    field="observer attention_entropy",
                )
                if value.attention_entropy is not None
                else None
            ),
        )
    if isinstance(value, Mapping):
        if "pav" in value:
            pav = value["pav"]
        elif "pav_variance" in value:
            pav = value["pav_variance"]
        else:
            raise ValueError("observer feature mapping requires pav")
        return ObserverFeatures(
            pav=_finite_vector(pav, field="observer pav"),
            layer24_mean=(
                _finite_vector(value["layer24_mean"], field="observer layer24_mean")
                if value.get("layer24_mean") is not None
                else None
            ),
            attention_entropy=(
                _finite_vector(
                    value["attention_entropy"],
                    field="observer attention_entropy",
                )
                if value.get("attention_entropy") is not None
                else None
            ),
        )
    return ObserverFeatures(pav=_finite_vector(value, field="observer pav"))


def _copy_observer_features(value: ObserverFeatures) -> ObserverFeatures:
    return ObserverFeatures(
        pav=value.pav.copy(),
        layer24_mean=(
            value.layer24_mean.copy() if value.layer24_mean is not None else None
        ),
        attention_entropy=(
            value.attention_entropy.copy()
            if value.attention_entropy is not None
            else None
        ),
    )


def _copy_cache_value(value: ObserverFeatures) -> ObserverFeatures | np.ndarray:
    copied = _copy_observer_features(value)
    if copied.layer24_mean is None and copied.attention_entropy is None:
        return copied.pav
    return copied


def _observer_features_equal(left: ObserverFeatures, right: ObserverFeatures) -> bool:
    for field in ("pav", "layer24_mean", "attention_entropy"):
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value is None or right_value is None:
            if left_value is not None or right_value is not None:
                return False
            continue
        if not np.array_equal(left_value, right_value):
            return False
    return True


def _validate_pav_matrix(pav_matrix: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(pav_matrix, dtype=float)
    target = np.asarray(labels)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("pav_matrix must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("pav_matrix must contain only finite values")
    if target.ndim != 1 or target.shape[0] != values.shape[0]:
        raise ValueError("labels must be one-dimensional with one entry per pAV row")
    if not np.all(np.isin(target, (0, 1))):
        raise ValueError("labels must contain only natural=0 and AI=1")
    if not np.any(target == 0) or not np.any(target == 1):
        raise ValueError("labels must contain both natural=0 and AI=1 examples")
    return values, target.astype(int, copy=False)


def _binary_auroc(values: np.ndarray, labels: np.ndarray) -> float:
    """Return AUROC using average ranks, including ties, without sklearn."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    n_ai = int(np.sum(labels == 1))
    n_natural = int(np.sum(labels == 0))
    return float((ranks[labels == 1].sum() - n_ai * (n_ai + 1) / 2.0) / (n_ai * n_natural))


def select_signed_heads(pav_matrix: Any, labels: Any, n_heads: int) -> SignedHeadSelection:
    """Select the most discriminative heads while retaining their AI direction.

    A positive sign means larger raw pAV values are AI-associated; a negative
    sign maps lower raw values toward positive AI evidence.  Centers and scales
    are calculated after signing, on natural examples only.
    """

    values, target = _validate_pav_matrix(pav_matrix, labels)
    if type(n_heads) is not int or not 1 <= n_heads <= values.shape[1]:
        raise ValueError("n_heads must be an integer between 1 and the number of pAV heads")

    aucs = np.array([_binary_auroc(values[:, head], target) for head in range(values.shape[1])])
    strengths = np.abs(aucs - 0.5)
    # Stable tie handling keeps artifacts reproducible: lower head index wins.
    ranking = np.lexsort((np.arange(values.shape[1]), -strengths))[:n_heads]
    signs = np.where(aucs[ranking] >= 0.5, 1.0, -1.0)
    natural_signed = values[target == 0][:, ranking] * signs
    centers = natural_signed.mean(axis=0)
    scales = natural_signed.std(axis=0)
    # A degenerate natural reference cannot standardize a head safely.  A unit
    # scale preserves a finite, explicitly centered feature rather than NaN.
    scales = np.where(scales > 0.0, scales, 1.0)
    return SignedHeadSelection(
        heads=tuple(int(head) for head in ranking),
        signs=tuple(float(sign) for sign in signs),
        centers=tuple(float(center) for center in centers),
        scales=tuple(float(scale) for scale in scales),
        aucs=tuple(float(aucs[head]) for head in ranking),
    )


def transform_signed_heads(
    values: Any,
    heads: Sequence[int],
    signs: Sequence[float],
    centers: Sequence[float],
    scales: Sequence[float],
) -> np.ndarray:
    """Select, sign, and natural-standardize pAV heads without collapsing them."""

    raw = np.asarray(values, dtype=float)
    original_ndim = raw.ndim
    if original_ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim != 2 or not np.all(np.isfinite(raw)):
        raise ValueError("values must be a finite one- or two-dimensional array")
    selected_heads = np.asarray(heads)
    selected_signs = np.asarray(signs, dtype=float)
    natural_centers = np.asarray(centers, dtype=float)
    natural_scales = np.asarray(scales, dtype=float)
    if selected_heads.ndim != 1 or not len(selected_heads):
        raise ValueError("heads must be a non-empty one-dimensional sequence")
    if not np.issubdtype(selected_heads.dtype, np.integer):
        raise ValueError("heads must contain integer indices")
    if np.any(selected_heads < 0) or np.any(selected_heads >= raw.shape[1]):
        raise ValueError("heads contains an out-of-range index")
    if len(set(int(item) for item in selected_heads)) != len(selected_heads):
        raise ValueError("heads must not contain duplicates")
    if any(vector.shape != selected_heads.shape for vector in (selected_signs, natural_centers, natural_scales)):
        raise ValueError("heads, signs, centers, and scales must have equal length")
    if not np.all(np.isin(selected_signs, (-1.0, 1.0))):
        raise ValueError("signs must be either -1.0 or 1.0")
    if not np.all(np.isfinite(natural_centers)) or not np.all(np.isfinite(natural_scales)):
        raise ValueError("centers and scales must be finite")
    if np.any(natural_scales <= 0.0):
        raise ValueError("scales must be positive")
    transformed = (raw[:, selected_heads] * selected_signs - natural_centers) / natural_scales
    return transformed[0] if original_ndim == 1 else transformed


def normalize_token_count_squared(values: Any, token_count: int) -> np.ndarray:
    """Normalize token-position pAV values by the square of valid token count."""

    normalized = np.asarray(values, dtype=float)
    if normalized.ndim != 1 or not np.all(np.isfinite(normalized)):
        raise ValueError("pAV values must be a finite one-dimensional array")
    if type(token_count) is not int or token_count < 1:
        raise ValueError("token_count must be a positive integer")
    return normalized / float(token_count * token_count)


def pav_cache_key(
    source_id: str,
    start: int,
    end: int,
    sequence: str,
    observer_id: str,
    schema: str,
) -> str:
    """Return a content-addressed pAV cache key for one source window."""

    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a non-empty string")
    if type(start) is not int or type(end) is not int or start < 0 or end < start:
        raise ValueError("start and end must be valid integer half-open coordinates")
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("sequence must be a non-empty string")
    if not isinstance(observer_id, str) or not observer_id:
        raise ValueError("observer_id must be a non-empty string")
    if not isinstance(schema, str) or not schema:
        raise ValueError("schema must be a non-empty string")
    record = {
        "end": end,
        "observer_id": observer_id,
        "schema": schema,
        "sequence_hash": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
        "source_id": source_id,
        "start": start,
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def limit_pav_candidate_windows(
    windows: Sequence[ScoringWindow], max_windows: int | None
) -> tuple[ScoringWindow, ...]:
    """Return at most max_windows deterministically spread pAV candidates."""

    materialized = tuple(windows)
    if type(max_windows) is not int or max_windows < 1:
        if max_windows is None:
            return materialized
        raise ValueError("max_windows must be a positive integer or None")
    if len(materialized) <= max_windows:
        return materialized
    indices = np.linspace(0, len(materialized) - 1, max_windows, dtype=int)
    return tuple(materialized[int(index)] for index in indices)


def _covered_window_bases(windows: Sequence[ScoringWindow]) -> int:
    spans = sorted((window.span for window in windows), key=lambda span: (span.start, span.end))
    if not spans:
        return 0
    total = 0
    start, end = spans[0].start, spans[0].end
    for span in spans[1:]:
        if span.start <= end:
            end = max(end, span.end)
        else:
            total += end - start
            start, end = span.start, span.end
    return total + end - start


class JsonlPAVCache:
    """Append-only JSONL cache for raw pAV or rich observer features."""

    def __init__(
        self,
        path: str | Path,
        *,
        log_every: int = 0,
        log_handle: Any = None,
    ) -> None:
        self.path = Path(path)
        if type(log_every) is not int or log_every < 0:
            raise ValueError("log_every must be a non-negative integer")
        self._log_every = log_every
        self._log_handle = log_handle if log_handle is not None else sys.stderr
        self._items: dict[str, ObserverFeatures] = {}
        self._writes = 0
        if self.path.exists():
            self._load_existing()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def _load_existing(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid pAV cache JSON at line {line_number}"
                    ) from error
                if not isinstance(record, Mapping):
                    raise ValueError(f"pAV cache line {line_number} must be an object")
                if record.get("schema") not in {
                    _PAV_CACHE_JSONL_SCHEMA,
                    _LEGACY_PAV_CACHE_JSONL_SCHEMA,
                }:
                    raise ValueError(f"pAV cache line {line_number} has an invalid schema")
                key = record.get("key")
                if not isinstance(key, str) or not key:
                    raise ValueError(f"pAV cache line {line_number} requires a key")
                self._items[key] = self._validate_value(
                    record.get("values"),
                    field=f"pAV cache line {line_number} values",
                )

    @staticmethod
    def _validate_vector(value: Any, *, field: str = "pAV cache value") -> np.ndarray:
        return _finite_vector(value, field=field)

    @staticmethod
    def _validate_value(value: Any, *, field: str = "pAV cache value") -> ObserverFeatures:
        try:
            return observer_features(value)
        except ValueError as error:
            raise ValueError(
                f"{field} must be a legacy pAV vector or feature mapping"
            ) from error

    @staticmethod
    def _json_value(value: ObserverFeatures) -> Any:
        if value.layer24_mean is None and value.attention_entropy is None:
            return value.pav.tolist()
        return {
            "attention_entropy": (
                value.attention_entropy.tolist()
                if value.attention_entropy is not None
                else None
            ),
            "layer24_mean": (
                value.layer24_mean.tolist()
                if value.layer24_mean is not None
                else None
            ),
            "pav": value.pav.tolist(),
        }

    def get(self, key: str, default: Any = None) -> Any:
        value = self._items.get(key)
        return default if value is None else _copy_cache_value(value)

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("pAV cache key must be a non-empty string")
        features = self._validate_value(value)
        existing = self._items.get(key)
        if existing is not None:
            if not _observer_features_equal(existing, features):
                raise ValueError("pAV cache key collision with different values")
            return
        self._items[key] = _copy_observer_features(features)
        json.dump(
            {
                "schema": _PAV_CACHE_JSONL_SCHEMA,
                "key": key,
                "values": self._json_value(features),
            },
            self._handle,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._handle.write("\n")
        self._handle.flush()
        self._writes += 1
        if self._log_every and self._writes % self._log_every == 0:
            print(
                f"pAV cache stored {self._writes} new observations "
                f"({len(self._items)} total)",
                file=self._log_handle,
                flush=True,
            )

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlPAVCache":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class NucleotideTransformerObserver:
    """Callable NT observer whose heavyweight dependencies load only on demand."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        torch_module: Any,
        device: Any,
        *,
        observer_config: ObserverConfig,
        observer_id: str | None = None,
    ) -> None:
        if not isinstance(observer_config, ObserverConfig):
            raise TypeError("observer_config must be an ObserverConfig")
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._device = device
        self._max_length = observer_config.max_length
        self._normalization = observer_config.normalization
        self.observer_config = observer_config
        self.observer_fingerprint = observer_config.fingerprint
        self.observer_id = observer_id or observer_config.model_name
        self.last_token_count: int | None = None

    @classmethod
    def load(
        cls,
        observer_config: ObserverConfig,
    ) -> "NucleotideTransformerObserver":
        """Load the production NT model lazily, after pAV was requested."""

        if not isinstance(observer_config, ObserverConfig):
            raise TypeError("observer_config must be an ObserverConfig")

        # Keep these imports here: CPU-only tests and normal v4 imports must not
        # require optional GPU/model packages or cause a model download.
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            observer_config.tokenizer_identity,
            revision=observer_config.revision,
        )
        model_kwargs = {
            "revision": observer_config.revision,
            "output_attentions": True,
            "output_hidden_states": True,
        }
        try:
            model = AutoModel.from_pretrained(
                observer_config.model_name,
                attn_implementation="eager",
                **model_kwargs,
            )
        except TypeError:
            model = AutoModel.from_pretrained(
                observer_config.model_name,
                **model_kwargs,
            )
        model.to(device).eval()
        return cls(
            tokenizer,
            model,
            torch,
            device,
            observer_config=observer_config,
            observer_id=observer_config.model_name,
        )

    def __call__(self, sequence: str) -> ObserverFeatures:
        inputs = self._tokenizer(
            sequence, return_tensors="pt", truncation=True, max_length=self._max_length
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._device)
        else:
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
            )
        mask_tensor = inputs["attention_mask"][0].detach().bool()
        mask = mask_tensor.cpu().numpy()
        token_count = int(mask.sum())
        if token_count < 1:
            raise ValueError("observer received no unmasked tokens")
        self.last_token_count = token_count
        distance = np.abs(np.arange(token_count)[:, None] - np.arange(token_count)[None, :])
        expected_distances: list[np.ndarray] = []
        if outputs.attentions is None:
            raise ValueError("observer did not return attentions")
        for layer_attention in outputs.attentions:
            attention = (
                layer_attention.squeeze(0)[:, :token_count, :token_count]
                .to(self._torch.float32)
                .detach()
                .cpu()
                .numpy()
            )
            expected_distances.extend(np.sum(head * distance, axis=1) for head in attention)
        pav = np.var(np.asarray(expected_distances), axis=1)
        if self._normalization == "token_count_squared":
            pav = normalize_token_count_squared(pav, token_count)
        if outputs.hidden_states is None:
            raise ValueError("observer did not return hidden states")
        if len(outputs.hidden_states) <= 24:
            raise ValueError("observer did not return layer 24 hidden states")
        hidden = outputs.hidden_states[24][0, mask_tensor].to(self._torch.float32)

        entropy_normalizer = math.log(float(token_count)) if token_count > 1 else 1.0
        entropy: list[np.ndarray] = []
        for layer_attention in outputs.attentions:
            attention = layer_attention[0, :, :token_count, :token_count].to(
                self._torch.float32
            )
            probabilities = self._torch.clamp(attention, min=1e-12)
            entropy.append(
                (
                    -self._torch.sum(
                        probabilities * self._torch.log(probabilities),
                        dim=-1,
                    ).mean(dim=-1)
                    / entropy_normalizer
                )
                .detach()
                .cpu()
                .numpy()
            )
        return ObserverFeatures(
            pav=np.asarray(pav, dtype=float),
            layer24_mean=np.asarray(
                hidden.mean(dim=0).detach().cpu().numpy(),
                dtype=float,
            ),
            attention_entropy=np.concatenate(entropy).astype(float, copy=False),
        )


def _artifact_error(message: str) -> ArtifactCompatibilityError:
    return ArtifactCompatibilityError(f"GOS v4 pAV artifact compatibility error: {message}")


def observer_configuration(
    *,
    model_name: str,
    revision: str,
    max_length: int,
    tokenizer_identity: str,
    normalization: str,
) -> ObserverConfig:
    """Validate and normalize every observer field that affects pAV values."""

    try:
        return ObserverConfig(
            model_name=model_name,
            revision=revision,
            tokenizer_identity=tokenizer_identity,
            max_length=max_length,
            normalization=normalization,
        )
    except ValueError as error:
        raise _artifact_error(str(error)) from error


def _observer_configuration(metadata: Mapping[str, Any]) -> ObserverConfig:
    raw = metadata.get("observer_config")
    try:
        config = ObserverConfig.from_mapping(raw)
    except (TypeError, ValueError) as error:
        raise _artifact_error(f"metadata {error}") from error
    fingerprint = metadata.get("observer_fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != config.fingerprint:
        raise _artifact_error(
            "metadata observer_fingerprint must match the canonical observer configuration"
        )
    return config


def observer_config_from_artifact(artifact: Mapping[str, Any]) -> ObserverConfig:
    """Read and verify the canonical observer configuration from a loaded artifact."""

    if not isinstance(artifact, Mapping):
        raise _artifact_error("artifact must be a mapping")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _artifact_error("artifact metadata must be a mapping")
    return _observer_configuration(metadata)


def observer_id_from_artifact(artifact: Mapping[str, Any]) -> str:
    """Read the non-authoritative display identity from a loaded pAV artifact."""

    if not isinstance(artifact, Mapping):
        raise _artifact_error("artifact must be a mapping")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _artifact_error("artifact metadata must be a mapping")
    value = metadata.get("observer_id", metadata.get("observer_identity"))
    if not isinstance(value, str) or not value:
        raise _artifact_error("metadata must include a non-empty observer_id")
    return value


def _observer_cache_identity(
    observer_id: str, config: ObserverConfig | Mapping[str, Any]
) -> str:
    """Return value identity; the display ID is validated but not hashed."""

    if not isinstance(observer_id, str) or not observer_id:
        raise ValueError("observer_id must be a non-empty string")
    if not isinstance(config, ObserverConfig):
        try:
            config = ObserverConfig.from_mapping(config)
        except (TypeError, ValueError) as error:
            raise ValueError("config must be a valid ObserverConfig") from error
    return config.fingerprint


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise _artifact_error(f"{field} must be a finite number")
    return float(value)


def _failure_warning(error: Exception, sequence: str) -> str:
    """Return compact diagnostic context while never exposing raw nucleotide data."""

    message = " ".join(str(error).split())
    if sequence:
        message = message.replace(sequence, "<sequence>")
    # A dependency may include only part of the original sequence in its
    # exception. Remove even short nucleotide runs: four bases are enough to
    # reveal a caller fragment yet uncommon enough to preserve normal prose.
    message = re.sub(r"[ACGT]{4,}", "<sequence>", message, flags=re.IGNORECASE)
    message = message[:160]
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


class PAVStage:
    """Score pAV windows independently and fail closed when coverage is inadequate."""

    def __init__(
        self,
        artifact: Mapping[str, Any],
        *,
        observer: Callable[[str], Any] | None = None,
        observer_id: str | None = None,
        cache: MutableMapping[str, Any] | None = None,
        allow_uncertified: bool = False,
    ) -> None:
        if type(allow_uncertified) is not bool:
            raise TypeError("allow_uncertified must be an exact boolean")
        artifact = validate_artifact(artifact, expected_kind=_PAV_ARTIFACT_KIND)
        self.certified = artifact_is_certified(artifact)
        if not self.certified and not allow_uncertified:
            raise _artifact_error(
                "uncertified pAV artifact requires allow_uncertified=True"
            )
        metadata = artifact["metadata"]
        payload = artifact["payload"]
        if not isinstance(metadata, Mapping) or not isinstance(payload, Mapping):
            raise _artifact_error("metadata and payload must be mappings")
        self._artifact = artifact
        self._validate_payload_keys(payload)
        self._heads, self._signs, self._centers, self._scales = self._selection(payload)
        expected_features = [f"pav_head_{head}" for head in self._heads]
        if artifact["feature_names"] != expected_features:
            raise _artifact_error(
                "feature_names must match selected head order "
                f"{expected_features!r}"
            )
        artifact_observer_id = self._observer_identity(metadata)
        self._observer_config = _observer_configuration(metadata)
        self._observer_fingerprint = self._observer_config.fingerprint
        if observer is not None:
            if not isinstance(observer_id, str) or not observer_id:
                raise ValueError("injected observer requires a non-empty observer_id")
            if observer_id != artifact_observer_id:
                raise _artifact_error(
                    "injected observer identity does not match the artifact"
                )
            self._observer_id = artifact_observer_id
        elif observer_id is not None:
            if not isinstance(observer_id, str) or not observer_id:
                raise ValueError("observer_id must be a non-empty string")
            if observer_id != artifact_observer_id:
                raise _artifact_error("observer identity does not match the artifact")
            self._observer_id = artifact_observer_id
        else:
            self._observer_id = artifact_observer_id
        self._normalization = metadata.get("normalization", "none")
        if self._normalization not in _SUPPORTED_NORMALIZATIONS:
            raise _artifact_error("normalization must be 'none' or 'token_count_squared'")
        if self._normalization != self._observer_config.normalization:
            raise _artifact_error(
                "normalization must match observer_config.normalization"
            )
        if observer is not None:
            declared_config = getattr(observer, "observer_config", None)
            if not isinstance(declared_config, ObserverConfig):
                raise _artifact_error(
                    "injected observer configuration must be an ObserverConfig"
                )
            if declared_config != self._observer_config:
                raise _artifact_error(
                    "injected observer configuration does not match the artifact"
                )
            declared_fingerprint = getattr(observer, "observer_fingerprint", None)
            if not isinstance(declared_fingerprint, str):
                raise _artifact_error(
                    "injected observer must expose its canonical observer fingerprint"
                )
            if (
                declared_fingerprint != declared_config.fingerprint
                or declared_fingerprint != self._observer_fingerprint
            ):
                raise _artifact_error(
                    "injected observer fingerprint does not match the artifact"
                )
        self._min_scored_coverage = _finite_number(
            metadata.get("min_scored_coverage", 1.0), "min_scored_coverage"
        )
        if not 0.0 < self._min_scored_coverage <= 1.0:
            raise _artifact_error("min_scored_coverage must be in (0, 1]")
        thresholds = metadata.get("thresholds")
        calibration = (
            thresholds.get("calibration") if isinstance(thresholds, Mapping) else None
        )
        if not isinstance(calibration, Mapping):
            raise _artifact_error("metadata must include pAV calibration metadata")
        self._window_threshold = _finite_number(
            calibration.get("threshold"), "calibrated window threshold"
        )
        if not 0.0 <= self._window_threshold <= 1.0:
            raise _artifact_error("window_threshold must be between 0 and 1")
        self._estimator = payload["estimator"]
        self._scaler = payload["scaler"]
        if not callable(getattr(self._estimator, "predict_proba", None)):
            raise _artifact_error("payload estimator must provide predict_proba")
        if not callable(getattr(self._scaler, "transform", None)):
            raise _artifact_error("payload scaler must provide transform")
        self._clip_lower = self._clip_vector(payload["clip_lower"], "clip_lower")
        self._clip_upper = self._clip_vector(payload["clip_upper"], "clip_upper")
        if np.any(self._clip_lower > self._clip_upper):
            raise _artifact_error("clip_lower must not exceed clip_upper")
        self._observer = observer
        self._cache = cache if cache is not None else {}
        self._max_windows_per_record = self._metadata_window_limit(
            metadata.get("max_windows_per_record")
        )

    @staticmethod
    def _validate_payload_keys(payload: Mapping[str, Any]) -> None:
        missing = sorted(_REQUIRED_PAYLOAD_KEYS - set(payload))
        if missing:
            raise _artifact_error(f"payload missing required keys: {', '.join(missing)}")
        legacy = sorted(_LEGACY_PAYLOAD_KEYS & set(payload))
        if legacy:
            raise _artifact_error(f"payload contains unsupported legacy keys: {', '.join(legacy)}")

    def _clip_vector(self, value: Any, field: str) -> np.ndarray:
        try:
            vector = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise _artifact_error(f"{field} must be a finite numeric vector") from error
        if vector.shape != (len(self._heads),) or not np.all(np.isfinite(vector)):
            raise _artifact_error(
                f"{field} must be a finite numeric vector matching selected_heads"
            )
        return vector

    @staticmethod
    def _selection(payload: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        try:
            heads = tuple(payload["selected_heads"])
            signs = tuple(float(value) for value in payload["signs"])
            centers = tuple(float(value) for value in payload["centers"])
            scales = tuple(float(value) for value in payload["scales"])
        except (TypeError, ValueError) as error:
            raise _artifact_error("selected_heads and signed scaling fields must be sequences") from error
        if not heads or any(type(head) is not int or head < 0 for head in heads):
            raise _artifact_error("heads must be non-negative integer indices")
        if len(set(heads)) != len(heads):
            raise _artifact_error("heads must not contain duplicates")
        if not all(len(items) == len(heads) for items in (signs, centers, scales)):
            raise _artifact_error("heads, signs, centers, and scales must have equal length")
        if not all(sign in {-1.0, 1.0} for sign in signs):
            raise _artifact_error("signs must be -1.0 or 1.0")
        if not all(np.isfinite(value) for value in centers) or not all(
            np.isfinite(value) and value > 0.0 for value in scales
        ):
            raise _artifact_error("centers must be finite and scales must be positive finite values")
        return heads, signs, centers, scales

    @staticmethod
    def _observer_identity(metadata: Mapping[str, Any]) -> str:
        return observer_id_from_artifact({"metadata": metadata})

    @staticmethod
    def _metadata_window_limit(value: Any) -> int | None:
        if value is None:
            return None
        if type(value) is not int or value < 1:
            raise _artifact_error("max_windows_per_record must be a positive integer")
        return value

    def _get_observer(self) -> Callable[[str], Any]:
        if self._observer is None:
            self._observer = NucleotideTransformerObserver.load(self._observer_config)
        return self._observer

    def _observe(
        self, window: ScoringWindow, source_id: str
    ) -> tuple[np.ndarray, ObserverFeatures]:
        cache_key = pav_cache_key(
            source_id,
            window.span.start,
            window.span.end,
            window.sequence,
            self._observer_fingerprint,
            _PAV_CACHE_SCHEMA,
        )
        cached = self._cache.get(cache_key)
        if cached is None:
            cached = observer_features(self._get_observer()(window.sequence))
            self._cache[cache_key] = cached
        else:
            cached = observer_features(cached)
        transformed = transform_signed_heads(
            cached.pav,
            self._heads,
            self._signs,
            self._centers,
            self._scales,
        )
        return transformed, cached

    def _window_score(self, transformed: np.ndarray) -> float:
        features = np.clip(transformed, self._clip_lower, self._clip_upper).reshape(1, -1)
        features = np.asarray(self._scaler.transform(features), dtype=float)
        if features.shape != (1, len(self._heads)) or not np.all(np.isfinite(features)):
            raise ValueError("pAV scaler returned invalid feature values")
        probabilities = np.asarray(self._estimator.predict_proba(features), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise ValueError("pAV estimator returned invalid probabilities")
        classes = getattr(self._estimator, "classes_", None)
        index = 1
        if classes is not None:
            matches = np.flatnonzero(np.asarray(classes) == 1)
            if len(matches) != 1:
                raise ValueError("pAV estimator must expose one AI class labelled 1")
            index = int(matches[0])
        if probabilities.shape[1] <= index:
            raise ValueError("pAV estimator is missing the AI probability column")
        score = float(probabilities[0, index])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("pAV estimator returned an invalid AI probability")
        return score

    @staticmethod
    def _aggregate(
        windows: Sequence[ScoringWindow],
        successes: Sequence[tuple[ScoringWindow, np.ndarray, ObserverFeatures, float]],
        *,
        total_eligible_bases: int,
    ) -> PAVAggregates:
        scored_bases = _covered_window_bases([window for window, _, _, _ in successes])
        coverage = scored_bases / total_eligible_bases if total_eligible_bases else 0.0
        transformed = np.asarray([values for _, values, _, _ in successes], dtype=float)
        transformed_mean = (
            tuple(float(value) for value in transformed.mean(axis=0)) if len(successes) else None
        )
        layer24_mean = _mean_optional_vectors(
            tuple(features.layer24_mean for _, _, features, _ in successes)
        )
        attention_entropy = _mean_optional_vectors(
            tuple(features.attention_entropy for _, _, features, _ in successes)
        )
        scores = [score for _, _, _, score in successes]
        return PAVAggregates(
            available=bool(successes),
            n_windows=len(windows),
            n_scored=len(successes),
            scored_coverage=float(coverage),
            mean=float(np.mean(scores)) if scores else None,
            maximum=float(np.max(scores)) if scores else None,
            transformed_mean=transformed_mean,
            layer24_mean=layer24_mean,
            attention_entropy=attention_entropy,
        )

    def score(self, prepared: PreparedSequence, *, source_id: str = "unknown") -> StageEvidence:
        """Score each eligible window, returning unavailable/failed evidence on gaps.

        This stage never maps a missing observer, a failed window, or weak pAV
        evidence to a natural label.  Routing owns the eventual tri-state call.
        """

        if not isinstance(prepared, PreparedSequence):
            raise TypeError("prepared must be a PreparedSequence")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        windows = limit_pav_candidate_windows(
            pav_candidate_windows(prepared),
            self._max_windows_per_record,
        )
        if not windows:
            return StageEvidence(
                status=StageStatus.UNAVAILABLE,
                reason="no_scorable_windows",
                aggregates=PAVAggregates(False, 0, 0, 0.0, None, None, None),
            )

        successes: list[tuple[ScoringWindow, np.ndarray, ObserverFeatures, float]] = []
        evidence_windows: list[WindowEvidence] = []
        failures = 0
        failure_warnings: list[str] = []
        for window in windows:
            try:
                transformed, features = self._observe(window, source_id)
                score = self._window_score(transformed)
            except Exception as error:
                failures += 1
                failure_warnings.append(_failure_warning(error, window.sequence))
                evidence_windows.append(WindowEvidence(window.span, None, eligible=False, length_bin=window.length_bin))
                continue
            successes.append((window, transformed, features, score))
            evidence_windows.append(
                WindowEvidence(
                    window.span,
                    score,
                    over_threshold=score >= self._window_threshold,
                    eligible=True,
                    length_bin=window.length_bin,
                )
            )

        eligible_bases = _covered_window_bases(windows)
        aggregates = self._aggregate(
            windows, successes, total_eligible_bases=eligible_bases
        )
        if aggregates.scored_coverage < self._min_scored_coverage:
            return StageEvidence(
                status=StageStatus.FAILED if failures else StageStatus.UNAVAILABLE,
                reason="insufficient_scored_coverage",
                aggregates=aggregates,
                windows=tuple(evidence_windows),
                n_scored=aggregates.n_scored,
                n_failed=failures,
                warnings=tuple(failure_warnings + ["pav_scored_coverage_below_minimum"]),
            )
        return StageEvidence(
            status=StageStatus.OK,
            aggregates=aggregates,
            windows=tuple(evidence_windows),
            n_scored=aggregates.n_scored,
            n_failed=failures,
            warnings=tuple(failure_warnings),
        )


def _mean_optional_vectors(
    vectors: Sequence[np.ndarray | None],
) -> tuple[float, ...] | None:
    available = [vector for vector in vectors if vector is not None]
    if not available:
        return None
    matrix = np.vstack(available)
    return tuple(float(value) for value in matrix.mean(axis=0))
