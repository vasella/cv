# cv

## MP4 transcript extraction script

This repository includes `extract_transcript.py`, a Python script that:

- reads a local MP4 file,
- extracts audio with `ffmpeg`,
- transcribes speech using `faster-whisper`,
- optionally labels two speakers with `pyannote.audio` diarization.

### Install dependencies

```bash
pip install faster-whisper
# Optional (for speaker diarization)
pip install pyannote.audio
```

Also make sure `ffmpeg` is installed and available in your `PATH`.

### Usage

```bash
python extract_transcript.py input.mp4 --output transcript.txt
```

With optional diarization between 2 speakers:

```bash
export HF_TOKEN=your_huggingface_token
python extract_transcript.py input.mp4 --output transcript.txt --diarize
```

Optional flags:

- `--model small` (default: `small`)
- `--language en` (force language)
- `--hf-token <token>` (alternative to `HF_TOKEN` env var)
- `--check-env` (verify local dependencies without running transcription)

### Output format

```text
[00:00:01.000 - 00:00:03.500] SPEAKER_00: Hello everyone.
[00:00:03.700 - 00:00:05.100] SPEAKER_01: Hi!
```

## How to test quickly

### 1) Verify dependencies only

```bash
python extract_transcript.py --check-env
python extract_transcript.py --check-env --diarize
```

### 2) Create a tiny local MP4 test file

This command generates a 5-second sample video with a simple tone:

```bash
ffmpeg -y \
  -f lavfi -i testsrc=size=640x360:rate=25 \
  -f lavfi -i sine=frequency=880:sample_rate=16000 \
  -t 5 \
  -c:v libx264 -pix_fmt yuv420p \
  -c:a aac \
  sample.mp4
```

### 3) Run transcription end-to-end

```bash
python extract_transcript.py sample.mp4 --output sample_transcript.txt
cat sample_transcript.txt
```

If diarization dependencies/token are configured:

```bash
export HF_TOKEN=your_huggingface_token
python extract_transcript.py sample.mp4 --output sample_transcript_diarized.txt --diarize
cat sample_transcript_diarized.txt
```
