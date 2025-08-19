#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./setup_whisper_env.sh [--model base.en|small|medium|large-v3] [--prefix /path]
#
# Defaults:
#   --model  base.en
#   --prefix $HOME/.local/src/whisper.cpp

MODEL="base.en"
PREFIX="${HOME}/.local/src/whisper.cpp"

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)  MODEL="${2:-base.en}"; shift 2;;
    --prefix) PREFIX="${2:-${HOME}/.local/src/whisper.cpp}"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

info() { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*" >&2; }

need_cmd() { command -v "$1" >/dev/null 2>&1; }

detect_pm() {
  if need_cmd apt-get; then echo apt
  elif need_cmd dnf; then echo dnf
  elif need_cmd yum; then echo yum
  elif need_cmd zypper; then echo zypper
  elif need_cmd pacman; then echo pacman
  else echo ""
  fi
}

install_pkgs() {
  local pm="$1"
  case "$pm" in
    apt)
      sudo apt-get update -y
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ffmpeg git build-essential pkg-config golang make
      ;;
    dnf)
      sudo dnf install -y ffmpeg git gcc gcc-c++ make pkgconfig golang
      ;;
    yum)
      sudo yum install -y epel-release || true
      sudo yum install -y ffmpeg git gcc gcc-c++ make pkgconfig golang || \
        warn "If ffmpeg is missing, enable RPM Fusion: https://rpmfusion.org/Configuration"
      ;;
    zypper)
      sudo zypper refresh
      sudo zypper install -y ffmpeg git gcc gcc-c++ make pkg-config go
      ;;
    pacman)
      sudo pacman -Sy --noconfirm ffmpeg git base-devel pkgconf go
      ;;
    *)
      warn "Unsupported package manager. Please install: ffmpeg git gcc g++ make pkg-config go"
      ;;
  esac
}

# --- ensure deps ---
PM=$(detect_pm)
for c in git ffmpeg gcc make go; do
  if ! need_cmd "$c"; then
    if [[ -n "$PM" ]]; then
      info "Installing missing dependency: $c"
      install_pkgs "$PM"
      break
    else
      err "Missing '$c' and no known package manager found. Install deps and re-run."
      exit 1
    fi
  fi
done

# --- clone/update whisper.cpp ---
mkdir -p "$(dirname "$PREFIX")"
if [[ -d "$PREFIX/.git" ]]; then
  info "Updating existing whisper.cpp at $PREFIX"
  git -C "$PREFIX" pull --ff-only || warn "Could not fast-forward; continuing."
else
  info "Cloning whisper.cpp to $PREFIX"
  git clone https://github.com/ggerganov/whisper.cpp "$PREFIX"
fi

# --- build core ---
info "Building whisper.cpp core (this may take a moment)..."
make -C "$PREFIX" -j >/dev/null

# --- download model (ggml/gguf) ---
info "Downloading Whisper model: $MODEL"
if [[ -x "$PREFIX/models/download-ggml-model.sh" ]]; then
  bash "$PREFIX/models/download-ggml-model.sh" "$MODEL"
elif [[ -x "$PREFIX/models/download-gguf-model.sh" ]]; then
  bash "$PREFIX/models/download-gguf-model.sh" "$MODEL"
else
  err "No model download script found under $PREFIX/models/"
  exit 1
fi

# --- build Go bindings (no app yet) ---
info "Building Go bindings..."
make -C "$PREFIX/bindings/go" -j >/dev/null

# --- done ---
info "Done!"
echo "whisper.cpp root:  $PREFIX"
echo "Models directory:  $PREFIX/models"
echo "Go bindings:       $PREFIX/bindings/go"
echo
echo "Next steps:"
echo "  1) Create your Go app in its own folder."
echo "  2) Point it at a model under: $PREFIX/models"
echo "  3) Use the official Go bindings import path:"
echo "     github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
