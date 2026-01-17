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

### Automatic model download

OpenCaption can automatically download models when using `--model-dir`:

```bash
./opencaption -in "talk.mp4" \
  -model base.en \
  -model-dir ~/.local/share/opencaption/models
```

Available models: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`,
`medium`, `medium.en`, `large-v1`, `large-v2`, `large-v3`, `turbo`

## GPU Acceleration

OpenCaption supports GPU acceleration through whisper.cpp. GPU support is
determined at compile time based on how whisper.cpp was built.

### NVIDIA CUDA (Linux/Windows)

Build whisper.cpp with CUDA support:

```bash
cd whisper.cpp
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j
```

Then build OpenCaption:

```bash
WHISPER_CPP=/path/to/whisper.cpp make build
```

### Apple Metal (macOS)

Metal is enabled by default on macOS builds:

```bash
cd whisper.cpp
mkdir build && cd build
cmake .. -DGGML_METAL=ON
cmake --build . --config Release -j
```

### Vulkan (Cross-platform)

Build whisper.cpp with Vulkan support:

```bash
cd whisper.cpp
mkdir build && cd build
cmake .. -DGGML_VULKAN=ON
cmake --build . --config Release -j
```

### Using Device Selection

Use the `--device` flag to control compute device:

```bash
# Automatic selection (GPU if available, else CPU)
./opencaption -in "talk.mp4" -device auto

# Force CPU only
./opencaption -in "talk.mp4" -device cpu

# Request GPU (falls back to CPU if unavailable)
./opencaption -in "talk.mp4" -device gpu
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

## Configuration

### Config File

Create a config file (JSON or YAML) to store your preferred settings:

**config.yaml:**
```yaml
model: base.en
model_dir: ~/.local/share/opencaption/models
format: vtt
vad: true
window: 60
overlap: 1
max_chars: 42
max_lines: 2
```

**config.json:**
```json
{
  "model": "base.en",
  "model_dir": "~/.local/share/opencaption/models",
  "format": "vtt",
  "vad": true,
  "window": 60,
  "max_chars": 42
}
```

Use with:
```bash
./opencaption -config config.yaml -in "talk.mp4"
```

### Presets

Use built-in presets for common use cases:

```bash
# Fast transcription (uses tiny model)
./opencaption -preset fast -in "talk.mp4"

# Accurate transcription (uses large-v3 model)
./opencaption -preset accurate -in "talk.mp4"

# Subtitle format (SRT with shorter lines)
./opencaption -preset subtitle -in "talk.mp4"
```

## Batch Processing

Process all media files in a directory:

```bash
# Process all files in a directory
./opencaption -batch -in ./videos/ -out ./captions/

# Include subdirectories
./opencaption -batch -recursive -in ./videos/ -out ./captions/
```

## Output Formats

OpenCaption supports multiple output formats:

- **VTT** (WebVTT): Default format, suitable for web video
- **SRT** (SubRip): Common subtitle format
- **JSON**: Machine-readable format with metadata

```bash
# VTT output
./opencaption -in "talk.mp4" -format vtt -out "captions.vtt"

# SRT output
./opencaption -in "talk.mp4" -format srt -out "captions.srt"

# JSON output
./opencaption -in "talk.mp4" -format json -out "captions.json"
```

### JSON Output Structure

```json
{
  "version": "1.0",
  "format": "opencaption",
  "cue_count": 42,
  "duration_seconds": 180.5,
  "cues": [
    {
      "index": 1,
      "start": 0.0,
      "end": 2.5,
      "start_formatted": "00:00:00.000",
      "end_formatted": "00:00:02.500",
      "text": "Hello and welcome",
      "lines": ["Hello and welcome"]
    }
  ]
}
```

## CLI Reference

```
Usage: opencaption [options]

Input/Output:
  -in string          Input audio/video file or directory (for batch mode)
  -out string         Output captions file (.vtt, .srt, .json) or directory
  -format string      Caption format: vtt | srt | json (default "vtt")

Model:
  -model string       Path to ggml/gguf model or model name
  -model-dir string   Directory for model files (enables auto-download)
  -list-models        List available models with descriptions

Processing:
  -lang string        Language (e.g. 'en'); empty = auto
  -vad                Enable simple energy-based VAD (default true)
  -window int         Chunk window seconds (0 = whole file) (default 60)
  -overlap int        Chunk overlap seconds (default 1)
  -threads int        Threads (0 = auto)
  -device string      Compute device: auto | cpu | gpu (default "auto")

Formatting:
  -maxchars int       Max characters per line (default 42)
  -maxlines int       Max lines per cue (default 2)

Configuration:
  -config string      Path to config file (.json or .yaml)
  -preset string      Use preset: fast | accurate | subtitle

Batch Processing:
  -batch              Process all media files in input directory
  -recursive          Process subdirectories in batch mode
  -workers int        Parallel workers for batch mode (0 = auto)

Output Control:
  -verbose            Enable verbose/debug output
  -quiet              Suppress non-error output
  -version            Show version information

Other:
  -ffmpeg-path string Path to ffmpeg binary (default "ffmpeg")
```

## Features

### Modern CLI Features

- **Structured Logging**: Uses Go's `slog` for structured, leveled logging
- **Graceful Shutdown**: Handles SIGINT/SIGTERM for clean interruption
- **Progress Bar**: Visual progress for model downloads
- **Parallel Batch Processing**: Process multiple files concurrently
- **Version Information**: Build-time version injection with `-version`
- **Context Support**: Cancellation-aware processing throughout

### List Available Models

```bash
./opencaption -list-models
```

Output:
```
Available Whisper models:

NAME               SIZE       MULTILINGUAL DESCRIPTION
----------------------------------------------------------------------
tiny               ~75MB      Yes          Fastest, lowest accuracy
tiny.en            ~75MB      No           Fastest, English only
base               ~142MB     Yes          Fast, good accuracy
base.en            ~142MB     No           Fast, English only
small              ~466MB     Yes          Balanced speed/accuracy
small.en           ~466MB     No           Balanced, English only
medium             ~1.5GB     Yes          High accuracy
medium.en          ~1.5GB     No           High accuracy, English only
large-v3           ~3GB       Yes          Best accuracy
large-v3-turbo     ~1.6GB     Yes          Fast large model
```

## Development

### Running Tests

```bash
# Run tests without CGO (packages that don't require whisper.cpp)
CGO_ENABLED=0 go test -v ./internal/...

# Run with race detection
CGO_ENABLED=0 go test -race ./internal/config/... ./internal/batch/...
```

### Building with Version Information

```bash
go build -ldflags="-X opencaption/internal/version.Version=1.0.0 \
  -X opencaption/internal/version.Commit=$(git rev-parse --short HEAD) \
  -X opencaption/internal/version.Date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -o opencaption ./cmd/opencaption
```

### Code Quality

The project uses:
- `gofmt` for code formatting
- `golangci-lint` for comprehensive linting
- Race detection in tests
- Code coverage reporting

## License

MIT License - see [LICENSE](LICENSE) for details.
