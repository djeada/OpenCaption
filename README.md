# OpenCaption

OpenCaption is a CLI tool for generating captions from audio or video using
whisper.cpp.

## Requirements

- ffmpeg in PATH (or use `--ffmpeg-path`)
- Go 1.23+ for building
- whisper.cpp headers + libraries (see Model setup)

## Build

```bash
make build
```

If your whisper.cpp checkout is elsewhere, set `WHISPER_CPP`:

```bash
WHISPER_CPP=/path/to/whisper.cpp make build
```

## Model setup

```bash
make setup MODEL=base.en
```

## Example

```bash
./opencaption -in "talk.mp4" \
  -out "talk.vtt" \
  -model "$HOME/.local/src/whisper.cpp/models/ggml-base.en.bin" \
  -lang en \
  -window 60 -overlap 1 \
  -format vtt
```

Run via make (handles library paths):

```bash
make run RUN_ARGS='-in "talk.mp4" -out "talk.vtt" -model "$HOME/.local/src/whisper.cpp/models/ggml-base.en.bin"'
```

Write to stdout:

```bash
./opencaption -in "talk.mp4" -out - -format srt
```

Read from stdin:

```bash
cat audio.wav | ./opencaption -in - -out captions.vtt
```

Disable VAD:

```bash
./opencaption -in "talk.mp4" -out "talk.vtt" -vad=false
```
