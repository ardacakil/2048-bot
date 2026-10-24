"""Server settings from BOT2048_* environment variables (standard library only).

    BOT2048_CHECKPOINT   run name under checkpoints/ or a directory      (overnight-a0025)
    BOT2048_DEPTH        expectimax depth, 0 = greedy                     (2)
    BOT2048_QUEUE_CAP    games allowed to wait while one computes         (10)
    BOT2048_DELAYS_MS    playback delay per target; its keys are the only
                         valid targets                                    (2048:20,4096:20,8192:10,16384:5)
    BOT2048_BATCH_MS     flush a batch of moves at least this often       (50)
    BOT2048_BATCH_MOVES  ... or after this many moves                     (256)
    BOT2048_KEEPALIVE_S  queued clients get a message at least this often (15)
    BOT2048_MAX_MOVES    safety cap per game                              (200000)
    BOT2048_LOG_LEVEL    debug, info, warning, error                      (info)
    BOT2048_CORS_ORIGINS comma-separated origins for local frontend dev   (empty: same origin only)

BOT2048_HOST and BOT2048_PORT are read by the uvicorn command line (systemd unit), not here.
"""

import dataclasses
import os

DEFAULT_DELAYS_MS = {2048: 20, 4096: 20, 8192: 10, 16384: 5}
LOG_LEVELS = ("debug", "info", "warning", "error")


class ConfigError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Settings:
    checkpoint: str = "overnight-a0025"
    depth: int = 2
    queue_cap: int = 10
    delays_ms: dict = dataclasses.field(default_factory=lambda: dict(DEFAULT_DELAYS_MS))
    batch_ms: float = 50.0
    batch_moves: int = 256
    keepalive_s: float = 15.0
    max_moves: int = 200_000
    log_level: str = "info"
    cors_origins: tuple = ()
    lag_interval_s: float = 0.1  # event-loop lag sampling period for /api/stats (0 = off)

    @property
    def targets(self):
        return tuple(sorted(self.delays_ms))

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env

        def get(name, parse, default):
            raw = env.get(f"BOT2048_{name}", "").strip()
            if not raw:
                return default
            try:
                return parse(raw)
            except (ValueError, ConfigError) as e:
                raise ConfigError(f"BOT2048_{name}={raw!r}: {e}") from None

        d = cls()
        return cls(
            checkpoint=get("CHECKPOINT", str, d.checkpoint),
            depth=get("DEPTH", lambda v: _int(v, 0, 3), d.depth),
            queue_cap=get("QUEUE_CAP", lambda v: _int(v, 0, 1000), d.queue_cap),
            delays_ms=get("DELAYS_MS", parse_delays, d.delays_ms),
            batch_ms=get("BATCH_MS", lambda v: _float(v, 1, 10_000), d.batch_ms),
            batch_moves=get("BATCH_MOVES", lambda v: _int(v, 1, 100_000), d.batch_moves),
            keepalive_s=get("KEEPALIVE_S", lambda v: _float(v, 0.01, 90), d.keepalive_s),
            max_moves=get("MAX_MOVES", lambda v: _int(v, 1, 10_000_000), d.max_moves),
            log_level=get("LOG_LEVEL", _log_level, d.log_level),
            cors_origins=get("CORS_ORIGINS", lambda v: tuple(o.strip() for o in v.split(",") if o.strip()),
                             d.cors_origins),
        )


def _int(text, lo, hi):
    v = int(text)
    if not lo <= v <= hi:
        raise ConfigError(f"must be between {lo} and {hi}")
    return v


def _float(text, lo, hi):
    v = float(text)
    if not lo <= v <= hi:
        raise ConfigError(f"must be between {lo:g} and {hi:g}")
    return v


def _log_level(text):
    if text.lower() not in LOG_LEVELS:
        raise ConfigError(f"must be one of {', '.join(LOG_LEVELS)}")
    return text.lower()


def parse_delays(text):
    """'2048:20,4096:20' -> {2048: 20, 4096: 20} (tile: playback delay in ms)."""
    out = {}
    for item in text.split(","):
        tile, sep, ms = item.strip().partition(":")
        if not sep:
            raise ConfigError(f"expected tile:ms, got {item.strip()!r}")
        tile, ms = int(tile), int(ms)
        if tile < 4 or tile & (tile - 1):
            raise ConfigError(f"target {tile} is not a power of two >= 4")
        if not 0 <= ms <= 1000:
            raise ConfigError(f"delay {ms} ms must be between 0 and 1000")
        out[tile] = ms
    if not out:
        raise ConfigError("no targets")
    return out
