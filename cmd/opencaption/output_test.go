package main

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"

	"opencaption/internal/batch"
	"opencaption/internal/captions"
	"opencaption/internal/config"
)

func TestPrepareOutput(t *testing.T) {
	dir := t.TempDir()
	for _, explicitBatch := range []bool{true, false} {
		cfg := config.DefaultConfig()
		cfg.Input, cfg.BatchMode = dir, explicitBatch
		if err := prepareOutput(cfg); err != nil {
			t.Fatal(err)
		}
		if !cfg.BatchMode || cfg.Output != dir {
			t.Errorf("batch defaults not resolved: %+v", cfg)
		}
	}
	cfg := config.DefaultConfig()
	cfg.Input, cfg.Output = dir, filepath.Join(dir, "new", "captions")
	if err := prepareOutput(cfg); err != nil {
		t.Fatal(err)
	}
	if !batch.IsDirectory(cfg.Output) {
		t.Fatal("explicit output directory was not created")
	}
	if err := os.WriteFile(filepath.Join(dir, "Talk.MP4"), nil, 0644); err != nil {
		t.Fatal(err)
	}
	jobs, err := batch.ScanDirectory(cfg.Input, cfg.Output, cfg.Format, false)
	if err != nil || len(jobs) != 1 || jobs[0].Output != filepath.Join(cfg.Output, "Talk.vtt") {
		t.Fatalf("custom batch destination ignored: %+v, %v", jobs, err)
	}
	cfg = config.DefaultConfig()
	cfg.Input = "talk.mp4"
	if err := prepareOutput(cfg); err != nil || cfg.Output != "captions.vtt" || cfg.BatchMode {
		t.Fatalf("single-file defaults not resolved: %+v, %v", cfg, err)
	}
}

func TestPrepareOutputRejectsInvalidBatchPaths(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "existing-file")
	if err := os.WriteFile(file, []byte("keep me"), 0644); err != nil {
		t.Fatal(err)
	}
	for _, output := range []string{"-", file} {
		cfg := config.DefaultConfig()
		cfg.Input, cfg.Output = dir, output
		if err := prepareOutput(cfg); err == nil {
			t.Errorf("expected error for batch destination %q", output)
		}
	}
	cfg := config.DefaultConfig()
	cfg.BatchMode, cfg.Input = true, file
	if err := prepareOutput(cfg); err == nil {
		t.Fatal("batch input must be a directory")
	}
}

func TestWriteOutput(t *testing.T) {
	cues := []captions.Cue{{Idx: 1, End: time.Second, Lines: []string{"Hello"}, RawText: "Hello"}}
	for _, format := range []string{"vtt", "srt", "json"} {
		t.Run(format, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "nested", "captions."+format)
			if err := writeOutput(path, format, cues); err != nil {
				t.Fatal(err)
			}
			data, err := os.ReadFile(path)
			if err != nil || !strings.Contains(string(data), "Hello") {
				t.Fatalf("missing caption output: %q, %v", data, err)
			}
		})
	}
}

func TestWriteOutputPropagatesStdoutError(t *testing.T) {
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	r.Close()
	w.Close()
	original := os.Stdout
	os.Stdout = w
	defer func() { os.Stdout = original }()
	cues := []captions.Cue{{Idx: 1, End: time.Second, Lines: []string{"Hello"}}}
	for _, format := range []string{"vtt", "srt", "json"} {
		if err := writeOutput("-", format, cues); !errors.Is(err, os.ErrClosed) {
			t.Errorf("%s: got %v, want closed stdout error", format, err)
		}
	}
}

func TestBatchReturnsProcessingErrors(t *testing.T) {
	original := logger
	logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	defer func() { logger = original }()
	cfg := config.DefaultConfig()
	cfg.FFmpegPath = filepath.Join(t.TempDir(), "missing-ffmpeg")
	jobs := []batch.FileJob{{Input: "a.wav"}, {Input: "b.wav"}}
	processors := map[string]func(context.Context, *config.Config, whisper.Model, []batch.FileJob) error{
		"sequential": runBatchSequential,
		"parallel": func(ctx context.Context, cfg *config.Config, model whisper.Model, jobs []batch.FileJob) error {
			return runBatchParallel(ctx, cfg, []whisper.Model{model, model}, jobs)
		},
	}
	for name, process := range processors {
		t.Run(name, func(t *testing.T) {
			if err := process(context.Background(), cfg, nil, jobs); err == nil {
				t.Fatal("failed jobs must fail the batch")
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if err := process(ctx, cfg, nil, jobs); !errors.Is(err, context.Canceled) {
				t.Errorf("expected cancellation, got %v", err)
			}
		})
	}
}
