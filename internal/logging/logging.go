// Package logging provides structured logging using slog.
package logging

import (
	"context"
	"io"
	"log/slog"
	"os"
)

// Level represents the logging level.
type Level string

const (
	LevelDebug Level = "debug"
	LevelInfo  Level = "info"
	LevelWarn  Level = "warn"
	LevelError Level = "error"
)

// Options configures the logger.
type Options struct {
	Level  Level
	JSON   bool
	Output io.Writer
}

// DefaultOptions returns default logging options.
func DefaultOptions() Options {
	return Options{
		Level:  LevelInfo,
		JSON:   false,
		Output: os.Stderr,
	}
}

// Setup initializes the global logger with the given options.
func Setup(opts Options) *slog.Logger {
	var level slog.Level
	switch opts.Level {
	case LevelDebug:
		level = slog.LevelDebug
	case LevelWarn:
		level = slog.LevelWarn
	case LevelError:
		level = slog.LevelError
	default:
		level = slog.LevelInfo
	}

	handlerOpts := &slog.HandlerOptions{
		Level: level,
	}

	var handler slog.Handler
	if opts.JSON {
		handler = slog.NewJSONHandler(opts.Output, handlerOpts)
	} else {
		handler = slog.NewTextHandler(opts.Output, handlerOpts)
	}

	logger := slog.New(handler)
	slog.SetDefault(logger)
	return logger
}

// FromContext retrieves the logger from context, or returns the default logger.
func FromContext(ctx context.Context) *slog.Logger {
	if logger, ok := ctx.Value(loggerKey{}).(*slog.Logger); ok {
		return logger
	}
	return slog.Default()
}

// WithLogger adds a logger to the context.
func WithLogger(ctx context.Context, logger *slog.Logger) context.Context {
	return context.WithValue(ctx, loggerKey{}, logger)
}

type loggerKey struct{}

// Progress logs a progress message with percentage.
func Progress(logger *slog.Logger, msg string, current, total int) {
	pct := 0
	if total > 0 {
		pct = (current * 100) / total
	}
	logger.Info(msg,
		slog.Int("current", current),
		slog.Int("total", total),
		slog.Int("percent", pct),
	)
}
