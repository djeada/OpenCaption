# TODO

## Milestone 1: Repo Foundation (done)
- [x] Add `go.mod` and module layout.
- [x] Split `main` into packages under `internal/`.
- [x] Add `Makefile` with build/test/lint targets.
- [x] Expand `README.md` with build and usage.

## Milestone 2: Robust Decode + I/O (done)
- [x] Stream PCM from ffmpeg to avoid temp WAV files.
- [x] Add `--ffmpeg-path` and clearer dependency errors.
- [x] Support stdin/stdout (`-in -`, `-out -`) end-to-end.

## Milestone 3: Caption Quality (done)
- [x] Add basic VAD or silence trimming.
- [x] Improve overlap de-duplication with fuzzy matching.
- [x] Smarter cue splitting with punctuation awareness.

## Milestone 4: Performance + GPU (done)
- [x] Add `--device auto|cpu|gpu` with detection and fallback.
- [x] Document GPU build steps for whisper.cpp backends.
- [x] Add `--model-dir` and model auto-download.

## Milestone 5: CLI UX + Config (done)
- [x] Add config file support and presets.
- [x] Add batch mode for directories.
- [x] Add `json` output format.

## Milestone 6: Release Quality (done)
- [x] Unit tests for formatting and wrapping.
- [x] Integration tests with short audio fixtures.
- [x] CI for build/test and versioned releases.
