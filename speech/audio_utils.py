from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".flac",
    ".aac",
    ".wma",
}


def ensure_audio_path(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Expected a file path, got: {p}")
    return p


def is_supported_audio_file(
    path: str | Path,
    supported_extensions: Optional[Iterable[str]] = None,
) -> bool:
    p = Path(path)
    extensions = {ext.lower() for ext in (supported_extensions or DEFAULT_AUDIO_EXTENSIONS)}
    return p.suffix.lower() in extensions


def assert_supported_audio_file(
    path: str | Path,
    supported_extensions: Optional[Iterable[str]] = None,
) -> Path:
    p = ensure_audio_path(path)
    if not is_supported_audio_file(p, supported_extensions=supported_extensions):
        allowed = ", ".join(sorted(supported_extensions or DEFAULT_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio format: {p.suffix or '<none>'}. Allowed: {allowed}")
    return p


def get_audio_duration_seconds(path: str | Path) -> Optional[float]:
    p = ensure_audio_path(path)
    try:
        import soundfile as sf

        with sf.SoundFile(str(p)) as audio_file:
            if audio_file.samplerate <= 0:
                return None
            return float(len(audio_file)) / float(audio_file.samplerate)
    except Exception as exc:
        logger.debug("Could not read duration for %s: %s", p, exc)
        return None

