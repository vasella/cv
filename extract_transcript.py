#!/usr/bin/env python3
"""Transcribe a local MP4 file to text, with optional two-speaker diarization.

Examples:
  python extract_transcript.py input.mp4 --output transcript.txt
  python extract_transcript.py input.mp4 --output transcript.txt --diarize --hf-token YOUR_TOKEN
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract transcript from a local MP4 file into plain text."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Path to local .mp4 video",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("transcript.txt"),
        help="Path to output text file (default: transcript.txt)",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model size (tiny, base, small, medium, large-v3; default: small)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code for transcription (example: en, es)",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (tries to label SPEAKER_00/SPEAKER_01)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token for pyannote diarization (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check required dependencies and exit (does not transcribe).",
    )
    return parser.parse_args()


def check_ffmpeg() -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit(
            "Error: ffmpeg is required but not found. Install ffmpeg and retry."
        )


def check_environment(can_diarize: bool) -> int:
    failures: List[str] = []
    print("Environment check:")

    try:
        check_ffmpeg()
        print("- ffmpeg: OK")
    except SystemExit:
        failures.append("ffmpeg")
        print("- ffmpeg: MISSING")

    try:
        import faster_whisper  # noqa: F401
        print("- faster-whisper: OK")
    except ImportError:
        failures.append("faster-whisper")
        print("- faster-whisper: MISSING")

    if can_diarize:
        try:
            import pyannote.audio  # noqa: F401
            print("- pyannote.audio: OK")
        except ImportError:
            failures.append("pyannote.audio")
            print("- pyannote.audio: MISSING")

    if failures:
        print(f"Missing dependencies: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def extract_audio_to_wav(input_video: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{proc.stderr}")


def run_whisper(wav_path: Path, model: str, language: Optional[str]) -> List[Segment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit(
            "Missing dependency: faster-whisper. Install with:\n"
            "  pip install faster-whisper"
        )

    whisper_model = WhisperModel(model, compute_type="int8")
    segments_iter, _info = whisper_model.transcribe(
        str(wav_path),
        language=language,
        vad_filter=True,
    )

    segments: List[Segment] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append(Segment(start=seg.start, end=seg.end, text=text))
    return segments


def interval_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return max(0.0, end - start)


def diarize_segments(
    wav_path: Path,
    segments: List[Segment],
    hf_token: Optional[str],
) -> List[Segment]:
    if not segments:
        return segments
    if not hf_token:
        print(
            "Warning: --diarize enabled but no HF token provided. "
            "Returning transcript without speaker labels.",
            file=sys.stderr,
        )
        return segments

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print(
            "Warning: pyannote.audio is not installed. Install with:\n"
            "  pip install pyannote.audio\n"
            "Returning transcript without speaker labels.",
            file=sys.stderr,
        )
        return segments

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(str(wav_path), num_speakers=2)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Warning: diarization failed ({exc}). Returning transcript without speaker labels.",
            file=sys.stderr,
        )
        return segments

    speaker_regions: List[Tuple[float, float, str]] = []
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        speaker_regions.append((turn.start, turn.end, speaker))

    for seg in segments:
        best_speaker = None
        best_overlap = 0.0
        for start, end, speaker in speaker_regions:
            overlap = interval_overlap((seg.start, seg.end), (start, end))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        seg.speaker = best_speaker

    return segments


def format_timestamp(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def write_transcript(output_path: Path, segments: Iterable[Segment]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            speaker = seg.speaker if seg.speaker else "SPEAKER"
            f.write(
                f"[{format_timestamp(seg.start)} - {format_timestamp(seg.end)}] "
                f"{speaker}: {seg.text}\n"
            )


def main() -> None:
    args = parse_args()

    if args.check_env:
        raise SystemExit(check_environment(can_diarize=args.diarize))

    if args.input is None:
        sys.exit("Error: input file is required unless --check-env is used.")
    if not args.input.exists():
        sys.exit(f"Error: input file not found: {args.input}")
    if args.input.suffix.lower() != ".mp4":
        print("Warning: input is not .mp4; attempting anyway.", file=sys.stderr)

    check_ffmpeg()

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        extract_audio_to_wav(args.input, wav_path)

        segments = run_whisper(wav_path, model=args.model, language=args.language)

        if args.diarize:
            segments = diarize_segments(
                wav_path,
                segments,
                hf_token=args.hf_token,
            )

        write_transcript(args.output, segments)

    print(f"Transcript written to: {args.output}")


if __name__ == "__main__":
    main()
