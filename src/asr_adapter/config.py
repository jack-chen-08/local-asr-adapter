"""Configuration is environment-only.

The inference service address must never be hard-coded: the ticket requires that the
adapter point at whatever the customer's own cloud runs, and that a wrong/missing
address fails loudly instead of silently falling back to a public endpoint.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_CONCURRENCY = 2
DEFAULT_RETRIES = 2
DEFAULT_TIMEOUT_S = 120.0


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    base_url: str
    concurrency: int
    retries: int
    timeout_s: float
    sample_rate: int

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        env = dict(os.environ if env is None else env)
        raw = (env.get("IP_ASR_URL") or "").strip().rstrip("/")
        if not raw:
            raise ConfigError(
                "IP_ASR_URL is not set: point it at the local inference service, e.g. "
                "http://127.0.0.1:8809 (see docker-compose.yml)"
            )
        if not (raw.startswith("http://") or raw.startswith("https://")):
            raise ConfigError(f"IP_ASR_URL must be an http(s) URL, got {raw!r}")
        return cls(
            base_url=raw,
            concurrency=_positive_int(env.get("IP_ASR_CONCURRENCY"), DEFAULT_CONCURRENCY,
                                      "IP_ASR_CONCURRENCY"),
            retries=_non_negative_int(env.get("IP_ASR_RETRIES"), DEFAULT_RETRIES, "IP_ASR_RETRIES"),
            timeout_s=_positive_float(env.get("IP_ASR_TIMEOUT_S"), DEFAULT_TIMEOUT_S, "IP_ASR_TIMEOUT_S"),
            sample_rate=_positive_int(env.get("IP_ASR_SAMPLE_RATE"), 16000, "IP_ASR_SAMPLE_RATE"),
        )


def _positive_int(raw: str | None, default: int, name: str) -> int:
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")
    return value


def _non_negative_int(raw: str | None, default: int, name: str) -> int:
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ConfigError(f"{name} must be >= 0, got {value}")
    return value


def _positive_float(raw: str | None, default: float, name: str) -> float:
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")
    return value
