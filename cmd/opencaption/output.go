package main

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"opencaption/internal/batch"
	"opencaption/internal/captions"
	"opencaption/internal/config"
	"opencaption/internal/format"
)

// prepareOutput resolves mode-specific defaults and validates batch destinations
// before a model is loaded or downloaded.
func prepareOutput(cfg *config.Config) error {
	if cfg.BatchMode || batch.IsDirectory(cfg.Input) {
		if !batch.IsDirectory(cfg.Input) {
			return errors.New("batch mode requires -in to be a directory")
		}
		cfg.BatchMode = true
		if cfg.Output == "" {
			cfg.Output = cfg.Input
		}
		if cfg.Output == "-" {
			return errors.New("batch mode requires an output directory; stdout is only supported for single files")
		}
		if err := os.MkdirAll(cfg.Output, 0755); err != nil {
			return fmt.Errorf("create batch output directory: %w", err)
		}
	} else if cfg.Output == "" {
		cfg.Output = "captions.vtt"
	}
	return nil
}

func writeOutput(outPath, formatType string, cues []captions.Cue) (err error) {
	out := os.Stdout
	if outPath != "-" {
		if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
			return err
		}
		f, err := os.Create(outPath)
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, f.Close()) }()
		out = f
	}

	switch strings.ToLower(formatType) {
	case "json":
		return format.WriteJSON(out, cues)
	case "srt":
		return format.WriteSRT(out, cues)
	default:
		if !strings.HasSuffix(strings.ToLower(outPath), ".vtt") && outPath != "-" {
			slog.Warn("writing VTT format; consider using .vtt extension")
		}
		return format.WriteVTT(out, cues)
	}
}
