<div align="center">

# 🎬 OpenCaption

**Professional audio and video transcription powered by whisper.cpp**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Go Version](https://img.shields.io/badge/Go-1.23+-00ADD8?logo=go)](go.mod)
[![Built with whisper.cpp](https://img.shields.io/badge/whisper.cpp-powered-green)](https://github.com/ggerganov/whisper.cpp)

[Features](#-features) •
[Quick Start](#-quick-start) •
[Installation](#-installation) •
[GPU Acceleration](#-gpu-acceleration) •
[Examples](#-examples)

</div>

---

## 📖 About

OpenCaption is a powerful command-line tool that generates accurate captions and subtitles from audio and video files. Built on top of whisper.cpp, it offers fast, GPU-accelerated transcription with support for multiple output formats and batch processing capabilities.

## ✨ Features

- 🚀 **High Performance** - GPU acceleration via CUDA, Metal, or Vulkan
- 📦 **Multiple Formats** - Export to VTT, SRT, or JSON
- 🔄 **Batch Processing** - Process multiple files in parallel
- 🎯 **Smart Chunking** - Automatic VAD for optimal segmentation
- ⚙️ **Flexible Configuration** - Config files, presets, and CLI options
- 🌍 **Multi-language Support** - All whisper.cpp supported languages
- 📥 **Auto Download** - Automatic model downloading and management
- 💻 **Cross-platform** - Works on Linux, macOS, and Windows

## 🚀 Quick Start

```bash
# Download and install
make setup MODEL=base.en
make build

# Transcribe a video
./opencaption -in video.mp4 -out captions.vtt

# Use automatic model download
./opencaption -in video.mp4 -model base.en -model-dir ~/.opencaption/models
```

## 📋 Requirements

- **FFmpeg** - Must be in PATH or specify with `-ffmpeg-path`
- **Go 1.23+** - For building from source
- **whisper.cpp** - Headers and libraries (automatically handled by build process)

## 🔧 Installation

### Building from Source

```bash
# Basic build
make build
```

### Custom whisper.cpp Location

If your whisper.cpp checkout is in a non-standard location:

```bash
WHISPER_CPP=/path/to/whisper.cpp make build
```

### Model Setup

Download a whisper model:

```bash
make setup MODEL=base.en
```

### Available Models

| Model | Size | Multilingual | Best For |
|-------|------|--------------|----------|
| `tiny` | ~75MB | ✅ Yes | Quick testing, low resources |
| `tiny.en` | ~75MB | ❌ English only | Fast English transcription |
| `base` | ~142MB | ✅ Yes | Balanced speed/accuracy |
| `base.en` | ~142MB | ❌ English only | Recommended for English |
| `small` | ~466MB | ✅ Yes | Good accuracy, moderate speed |
| `small.en` | ~466MB | ❌ English only | High quality English |
| `medium` | ~1.5GB | ✅ Yes | High accuracy |
| `medium.en` | ~1.5GB | ❌ English only | Best English quality |
| `large-v3` | ~3GB | ✅ Yes | Maximum accuracy |
| `turbo` | ~1.6GB | ✅ Yes | Fast large model |

### Automatic Model Download

OpenCaption can automatically download models when needed:

```bash
./opencaption -in "talk.mp4" \
  -model base.en \
  -model-dir ~/.local/share/opencaption/models
```

---

## 🎮 GPU Acceleration

OpenCaption supports GPU acceleration for faster transcription. GPU support is determined at compile time based on your whisper.cpp build configuration.

### 🟢 NVIDIA CUDA (Linux/Windows)

1. Build whisper.cpp with CUDA:

```bash
cd whisper.cpp
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j
```

2. Build OpenCaption:

```bash
WHISPER_CPP=/path/to/whisper.cpp make build
```

### 🍎 Apple Metal (macOS)

1. Build whisper.cpp with Metal (default on macOS):

```bash
cd whisper.cpp
mkdir build && cd build
cmake .. -DGGML_METAL=ON
cmake --build . --config Release -j
```

2. Build OpenCaption as usual

### 🌋 Vulkan (Cross-platform)

1. Build whisper.cpp with Vulkan:

```bash
cd whisper.cpp
mkdir build && cd build
cmake .. -DGGML_VULKAN=ON
cmake --build . --config Release -j
```

2. Build OpenCaption as usual

### Device Selection

Control which compute device to use:

```bash
# Automatic selection (GPU if available, else CPU)
./opencaption -in "talk.mp4" -device auto

# Force CPU only
./opencaption -in "talk.mp4" -device cpu

# Request GPU (falls back to CPU if unavailable)
./opencaption -in "talk.mp4" -device gpu
```

---

## 📚 Examples

### Basic Usage

```bash
./opencaption -in "talk.mp4" \
  -out "talk.vtt" \
  -model "$HOME/.local/src/whisper.cpp/models/ggml-base.en.bin" \
  -lang en \
  -window 60 -overlap 1 \
  -format vtt
```

### Using Make (Handles Library Paths)

```bash
make run RUN_ARGS='-in "talk.mp4" -out "talk.vtt" -model "$HOME/.local/src/whisper.cpp/models/ggml-base.en.bin"'
```

### Write to stdout

```bash
./opencaption -in "talk.mp4" -out - -format srt
```

### Read from stdin

```bash
cat audio.wav | ./opencaption -in - -out captions.vtt
```

### Disable VAD

```bash
./opencaption -in "talk.mp4" -out "talk.vtt" -vad=false
```

---

## ⚙️ Configuration

### Configuration Files

Store your preferred settings in a config file (JSON or YAML):

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

### Built-in Presets

Quick configuration with presets:

```bash
# Fast transcription (uses tiny model)
./opencaption -preset fast -in "talk.mp4"

# Accurate transcription (uses large-v3 model)
./opencaption -preset accurate -in "talk.mp4"

# Subtitle format (SRT with shorter lines)
./opencaption -preset subtitle -in "talk.mp4"
```

---

## 📦 Batch Processing

Process multiple files efficiently:

```bash
# Process all files in a directory
./opencaption -batch -in ./videos/ -out ./captions/

# Include subdirectories
./opencaption -batch -recursive -in ./videos/ -out ./captions/
```

---

## 📄 Output Formats

OpenCaption supports multiple caption formats:

| Format | Extension | Use Case |
|--------|-----------|----------|
| **VTT** | `.vtt` | Web video (HTML5), default format |
| **SRT** | `.srt` | Common subtitle format, wide compatibility |
| **JSON** | `.json` | Machine-readable, includes metadata |

### Format Examples

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

---

## 🔍 CLI Reference

### Input/Output Options

| Flag | Type | Description |
|------|------|-------------|
| `-in` | string | Input audio/video file or directory (for batch mode) |
| `-out` | string | Output captions file (.vtt, .srt, .json) or directory |
| `-format` | string | Caption format: `vtt` \| `srt` \| `json` (default: `vtt`) |

### Model Options

| Flag | Type | Description |
|------|------|-------------|
| `-model` | string | Path to ggml/gguf model or model name |
| `-model-dir` | string | Directory for model files (enables auto-download) |
| `-list-models` | flag | List available models with descriptions |

### Processing Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-lang` | string | auto | Language code (e.g., 'en') |
| `-vad` | bool | `true` | Enable simple energy-based VAD |
| `-window` | int | `60` | Chunk window seconds (0 = whole file) |
| `-overlap` | int | `1` | Chunk overlap seconds |
| `-threads` | int | auto | Number of threads (0 = auto) |
| `-device` | string | `auto` | Compute device: `auto` \| `cpu` \| `gpu` |

### Formatting Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-maxchars` | int | `42` | Max characters per line |
| `-maxlines` | int | `2` | Max lines per cue |

### Configuration Options

| Flag | Type | Description |
|------|------|-------------|
| `-config` | string | Path to config file (.json or .yaml) |
| `-preset` | string | Use preset: `fast` \| `accurate` \| `subtitle` |

### Batch Processing Options

| Flag | Type | Description |
|------|------|-------------|
| `-batch` | flag | Process all media files in input directory |
| `-recursive` | flag | Process subdirectories in batch mode |
| `-workers` | int | Parallel workers for batch mode (0 = auto) |

### Output Control Options

| Flag | Description |
|------|-------------|
| `-verbose` | Enable verbose/debug output |
| `-quiet` | Suppress non-error output |
| `-version` | Show version information |

### Other Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-ffmpeg-path` | string | `ffmpeg` | Path to ffmpeg binary |

---

## 🛠️ Development

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

The project maintains high code quality standards:

- ✅ `gofmt` for consistent code formatting
- ✅ `golangci-lint` for comprehensive linting
- ✅ Race detection in tests
- ✅ Code coverage reporting

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [whisper.cpp](https://github.com/ggerganov/whisper.cpp) by Georgi Gerganov
- Powered by OpenAI's Whisper model

---

<div align="center">

**[⬆ Back to Top](#-opencaption)**

Made with ❤️ by [Adam Djellouli](https://github.com/djeada)

</div>
