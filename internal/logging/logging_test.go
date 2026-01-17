package logging

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
)

func TestDefaultOptions(t *testing.T) {
	opts := DefaultOptions()

	if opts.Level != LevelInfo {
		t.Errorf("Default Level = %q, want %q", opts.Level, LevelInfo)
	}
	if opts.JSON {
		t.Error("Default JSON should be false")
	}
	if opts.Output == nil {
		t.Error("Default Output should not be nil")
	}
}

func TestSetup(t *testing.T) {
	var buf bytes.Buffer
	opts := Options{
		Level:  LevelDebug,
		JSON:   false,
		Output: &buf,
	}

	logger := Setup(opts)

	logger.Info("test message", slog.String("key", "value"))

	output := buf.String()
	if !strings.Contains(output, "test message") {
		t.Errorf("Output should contain 'test message', got: %s", output)
	}
	if !strings.Contains(output, "key=value") {
		t.Errorf("Output should contain 'key=value', got: %s", output)
	}
}

func TestSetupJSON(t *testing.T) {
	var buf bytes.Buffer
	opts := Options{
		Level:  LevelInfo,
		JSON:   true,
		Output: &buf,
	}

	logger := Setup(opts)
	logger.Info("json test")

	output := buf.String()
	// JSON output should contain braces
	if !strings.Contains(output, "{") || !strings.Contains(output, "}") {
		t.Errorf("JSON output should contain braces, got: %s", output)
	}
}

func TestSetupLevels(t *testing.T) {
	tests := []struct {
		level     Level
		wantLevel slog.Level
	}{
		{LevelDebug, slog.LevelDebug},
		{LevelInfo, slog.LevelInfo},
		{LevelWarn, slog.LevelWarn},
		{LevelError, slog.LevelError},
		{"", slog.LevelInfo}, // default
	}

	for _, tt := range tests {
		t.Run(string(tt.level), func(t *testing.T) {
			var buf bytes.Buffer
			logger := Setup(Options{Level: tt.level, Output: &buf})

			// Logger should be created without panicking
			if logger == nil {
				t.Error("Setup should return non-nil logger")
			}
		})
	}
}

func TestFromContext(t *testing.T) {
	ctx := context.Background()

	// Without logger in context, should return default
	logger := FromContext(ctx)
	if logger == nil {
		t.Error("FromContext should return non-nil logger")
	}
}

func TestWithLogger(t *testing.T) {
	var buf bytes.Buffer
	customLogger := slog.New(slog.NewTextHandler(&buf, nil))

	ctx := context.Background()
	ctx = WithLogger(ctx, customLogger)

	retrieved := FromContext(ctx)
	if retrieved != customLogger {
		t.Error("FromContext should return the logger set with WithLogger")
	}
}

func TestProgress(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, nil))

	Progress(logger, "processing", 50, 100)

	output := buf.String()
	if !strings.Contains(output, "processing") {
		t.Errorf("Output should contain 'processing', got: %s", output)
	}
	if !strings.Contains(output, "percent=50") {
		t.Errorf("Output should contain 'percent=50', got: %s", output)
	}
}

func TestProgressZeroTotal(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, nil))

	// Should not panic with zero total
	Progress(logger, "test", 0, 0)

	output := buf.String()
	if !strings.Contains(output, "percent=0") {
		t.Errorf("Output should contain 'percent=0' when total is 0, got: %s", output)
	}
}
