from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from utils import Settings, get_settings
from .audio_utils import assert_supported_audio_file, get_audio_duration_seconds

logger = logging.getLogger(__name__)


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str


@dataclass
class ASRResult:
    text: str
    language: Optional[str] = None
    language_probability: Optional[float] = None
    segments: Optional[list[ASRSegment]] = None
    duration_seconds: Optional[float] = None


class ASRTranscriber:
    """Small, reusable ASR wrapper with CPU-friendly defaults."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        backend: Optional[str] = None,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        allow_backend_fallback: bool = True,
    ):
        self.settings = settings or get_settings()
        self.backend = (backend or self.settings.asr_backend).strip().lower()
        self.model_size = model_size or self.settings.whisper_model_size
        self.device = device or self.settings.asr_device
        self.compute_type = compute_type or self.settings.asr_compute_type
        self.allow_backend_fallback = allow_backend_fallback

        self._faster_whisper_model = None
        self._openai_whisper_model = None

    def transcribe(
        self,
        audio_path: str | Path,
        return_timestamps: bool = False,
        language: Optional[str] = None,
    ) -> ASRResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Local audio file path.
            return_timestamps: Include segment-level timestamps when backend supports them.
            language: Optional language hint (e.g., "en", "hi").
        """
        validated_path = assert_supported_audio_file(audio_path)
        duration_seconds = get_audio_duration_seconds(validated_path)

        errors: list[str] = []
        for backend in self._backend_attempt_order():
            try:
                if backend == "faster-whisper":
                    result = self._transcribe_faster_whisper(validated_path, return_timestamps, language)
                elif backend == "whisper":
                    result = self._transcribe_whisper(validated_path, return_timestamps, language)
                else:
                    raise ValueError(f"Unsupported ASR backend: {backend}")

                result.duration_seconds = duration_seconds
                return result
            except ImportError as exc:
                errors.append(f"{backend}: {exc}")
                logger.warning("ASR backend unavailable (%s): %s", backend, exc)

        if errors:
            joined = " | ".join(errors)
            raise RuntimeError(f"No ASR backend is available. Details: {joined}")

        raise RuntimeError("ASR transcription failed unexpectedly.")

    def _backend_attempt_order(self) -> list[str]:
        if self.backend not in {"faster-whisper", "whisper"}:
            raise ValueError(f"Unsupported ASR_BACKEND: {self.backend}")
        if not self.allow_backend_fallback:
            return [self.backend]
        fallback = "whisper" if self.backend == "faster-whisper" else "faster-whisper"
        return [self.backend, fallback]

    def _get_faster_whisper_model(self):
        if self._faster_whisper_model is not None:
            return self._faster_whisper_model

        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "faster-whisper is not available. Install it or switch backend to 'whisper'."
            ) from exc

        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            self.model_size,
            self.device,
            self.compute_type,
        )
        self._faster_whisper_model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        return self._faster_whisper_model

    def _get_openai_whisper_model(self):
        if self._openai_whisper_model is not None:
            return self._openai_whisper_model

        try:
            import whisper
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "openai-whisper is not installed. Install it manually (pip install openai-whisper) "
                "or switch backend to 'faster-whisper'."
            ) from exc

        logger.info("Loading openai-whisper model=%s device=%s", self.model_size, self.device)
        self._openai_whisper_model = whisper.load_model(self.model_size, device=self.device)
        return self._openai_whisper_model

    def _transcribe_faster_whisper(
        self,
        audio_path: Path,
        return_timestamps: bool,
        language: Optional[str],
    ) -> ASRResult:
        model = self._get_faster_whisper_model()
        segments_iter, info = model.transcribe(
            str(audio_path),
            task="transcribe",
            language=language,
            beam_size=1,
            vad_filter=True,
        )

        segments: Optional[list[ASRSegment]] = [] if return_timestamps else None
        chunks: list[str] = []
        for seg in segments_iter:
            text = (seg.text or "").strip()
            if not text:
                continue
            chunks.append(text)
            if segments is not None:
                segments.append(ASRSegment(start=float(seg.start), end=float(seg.end), text=text))

        return ASRResult(
            text=" ".join(chunks).strip(),
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            segments=segments,
        )

    def _transcribe_whisper(
        self,
        audio_path: Path,
        return_timestamps: bool,
        language: Optional[str],
    ) -> ASRResult:
        model = self._get_openai_whisper_model()
        result: dict[str, Any] = model.transcribe(
            str(audio_path),
            task="transcribe",
            language=language,
            fp16=False,
            verbose=False,
        )

        segments: Optional[list[ASRSegment]] = None
        if return_timestamps:
            segments = []
            for seg in result.get("segments", []):
                text = str(seg.get("text", "")).strip()
                if not text:
                    continue
                segments.append(
                    ASRSegment(
                        start=float(seg.get("start", 0.0)),
                        end=float(seg.get("end", 0.0)),
                        text=text,
                    )
                )

        return ASRResult(
            text=str(result.get("text", "")).strip(),
            language=result.get("language"),
            language_probability=None,
            segments=segments,
        )


if __name__ == "__main__":
    import argparse

    from utils.logger import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description="Local test runner for speech transcription.")
    parser.add_argument("--audio", required=True, help="Path to an audio file (wav/mp3/etc).")
    parser.add_argument(
        "--backend",
        default=None,
        choices=["faster-whisper", "whisper"],
        help="Override backend. Defaults to ASR_BACKEND from settings.",
    )
    parser.add_argument("--model", default=None, help="Model size, e.g. tiny, base, small.")
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Print segment-level timestamps when available.",
    )
    parser.add_argument("--language", default=None, help="Optional language hint like en, hi, fr.")
    args = parser.parse_args()

    transcriber = ASRTranscriber(backend=args.backend, model_size=args.model, device="cpu")
    result = transcriber.transcribe(
        args.audio,
        return_timestamps=args.timestamps,
        language=args.language,
    )

    print("\n=== Transcription ===")
    print(result.text)
    print(f"\nLanguage: {result.language}")
    if result.language_probability is not None:
        print(f"Language confidence: {result.language_probability:.3f}")
    if result.duration_seconds is not None:
        print(f"Audio duration (s): {result.duration_seconds:.2f}")

    if args.timestamps and result.segments:
        print("\n=== Segments ===")
        for seg in result.segments:
            print(f"[{seg.start:7.2f}s -> {seg.end:7.2f}s] {seg.text}")
