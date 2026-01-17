package models

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/schollz/progressbar/v3"
)

// Known models with their download URLs (Hugging Face mirrors)
var KnownModels = map[string]string{
	"tiny":             "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
	"tiny.en":          "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
	"base":             "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
	"base.en":          "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
	"small":            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
	"small.en":         "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
	"medium":           "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
	"medium.en":        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin",
	"large-v1":         "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin",
	"large-v2":         "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin",
	"large-v3":         "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
	"large":            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
	"large-v3-turbo":   "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
	"distil-medium.en": "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin",
	"distil-large-v2":  "https://huggingface.co/distil-whisper/distil-large-v2/resolve/main/ggml-large-32-2.bin",
}

// ModelInfo contains information about a model.
type ModelInfo struct {
	Name         string
	Size         string
	Description  string
	Multilingual bool
}

// ModelCatalog provides model information.
var ModelCatalog = map[string]ModelInfo{
	"tiny":           {Name: "tiny", Size: "~75MB", Description: "Fastest, lowest accuracy", Multilingual: true},
	"tiny.en":        {Name: "tiny.en", Size: "~75MB", Description: "Fastest, English only", Multilingual: false},
	"base":           {Name: "base", Size: "~142MB", Description: "Fast, good accuracy", Multilingual: true},
	"base.en":        {Name: "base.en", Size: "~142MB", Description: "Fast, English only", Multilingual: false},
	"small":          {Name: "small", Size: "~466MB", Description: "Balanced speed/accuracy", Multilingual: true},
	"small.en":       {Name: "small.en", Size: "~466MB", Description: "Balanced, English only", Multilingual: false},
	"medium":         {Name: "medium", Size: "~1.5GB", Description: "High accuracy", Multilingual: true},
	"medium.en":      {Name: "medium.en", Size: "~1.5GB", Description: "High accuracy, English only", Multilingual: false},
	"large-v3":       {Name: "large-v3", Size: "~3GB", Description: "Best accuracy", Multilingual: true},
	"large-v3-turbo": {Name: "large-v3-turbo", Size: "~1.6GB", Description: "Fast large model", Multilingual: true},
}

// DefaultModelDir returns the default model directory path.
func DefaultModelDir() string {
	// Check XDG_DATA_HOME first (modern standard)
	if xdgData := os.Getenv("XDG_DATA_HOME"); xdgData != "" {
		return filepath.Join(xdgData, "opencaption", "models")
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "./models"
	}
	return filepath.Join(home, ".local", "share", "opencaption", "models")
}

// ResolveModel resolves a model name or path to an actual file path.
// If modelDir is provided and the model doesn't exist, it will attempt to download it.
func ResolveModel(model, modelDir string, autoDownload bool) (string, error) {
	return ResolveModelWithContext(context.Background(), model, modelDir, autoDownload)
}

// ResolveModelWithContext resolves a model with context for cancellation.
func ResolveModelWithContext(ctx context.Context, model, modelDir string, autoDownload bool) (string, error) {
	// If model is an absolute path or exists, use it directly
	if filepath.IsAbs(model) {
		if _, err := os.Stat(model); err == nil {
			return model, nil
		}
		return "", fmt.Errorf("model file not found: %s", model)
	}

	// Check if model exists in current directory
	if _, err := os.Stat(model); err == nil {
		return model, nil
	}

	// Check if model exists in modelDir
	if modelDir != "" {
		// Try exact path
		modelPath := filepath.Join(modelDir, model)
		if _, err := os.Stat(modelPath); err == nil {
			return modelPath, nil
		}

		// Try with ggml- prefix
		if !strings.HasPrefix(filepath.Base(model), "ggml-") {
			modelPath = filepath.Join(modelDir, "ggml-"+model+".bin")
			if _, err := os.Stat(modelPath); err == nil {
				return modelPath, nil
			}
		}

		// If autoDownload is enabled, try to download
		if autoDownload {
			return downloadModelWithContext(ctx, model, modelDir)
		}
	}

	return "", fmt.Errorf("model not found: %s (use --model-dir or provide full path)", model)
}

// downloadModelWithContext downloads a model with context support.
func downloadModelWithContext(ctx context.Context, model, modelDir string) (string, error) {
	// Normalize model name
	name := strings.TrimSuffix(filepath.Base(model), ".bin")
	name = strings.TrimPrefix(name, "ggml-")

	url, ok := KnownModels[name]
	if !ok {
		return "", fmt.Errorf("unknown model: %s\nAvailable models: %s", name, ListKnownModels())
	}

	// Create model directory
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return "", fmt.Errorf("create model dir: %w", err)
	}

	// Destination path
	destName := "ggml-" + name + ".bin"
	destPath := filepath.Join(modelDir, destName)

	// Check if already exists
	if _, err := os.Stat(destPath); err == nil {
		return destPath, nil
	}

	// Create HTTP request with context
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	// Use a client without overall timeout - context handles cancellation
	// Large models (3GB+) can take over an hour on slow connections
	client := &http.Client{}

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("download model: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download failed: HTTP %d", resp.StatusCode)
	}

	// Create temp file
	tmpPath := destPath + ".tmp"
	out, err := os.Create(tmpPath)
	if err != nil {
		return "", fmt.Errorf("create temp file: %w", err)
	}

	// Create progress bar
	bar := progressbar.NewOptions64(
		resp.ContentLength,
		progressbar.OptionSetDescription(fmt.Sprintf("Downloading %s", name)),
		progressbar.OptionSetWriter(os.Stderr),
		progressbar.OptionShowBytes(true),
		progressbar.OptionSetWidth(40),
		progressbar.OptionThrottle(100*time.Millisecond),
		progressbar.OptionShowCount(),
		progressbar.OptionOnCompletion(func() {
			fmt.Fprint(os.Stderr, "\n")
		}),
		progressbar.OptionSpinnerType(14),
		progressbar.OptionFullWidth(),
		progressbar.OptionSetRenderBlankState(true),
	)

	// Copy with progress
	written, copyErr := io.Copy(io.MultiWriter(out, bar), resp.Body)
	out.Close()

	if copyErr != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("download model: %w", copyErr)
	}

	// Rename to final path
	if err := os.Rename(tmpPath, destPath); err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("rename model file: %w", err)
	}

	fmt.Fprintf(os.Stderr, "Downloaded %s (%.1f MB)\n", destName, float64(written)/1024/1024)
	return destPath, nil
}

// ListKnownModels returns a formatted list of known models.
func ListKnownModels() string {
	var names []string
	for k := range KnownModels {
		names = append(names, k)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

// ListAvailable returns available models in the model directory.
func ListAvailable(modelDir string) ([]string, error) {
	if modelDir == "" {
		modelDir = DefaultModelDir()
	}

	entries, err := os.ReadDir(modelDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var models []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, ".bin") && strings.HasPrefix(name, "ggml-") {
			models = append(models, name)
		}
	}
	sort.Strings(models)
	return models, nil
}

// GetModelInfo returns information about a model.
func GetModelInfo(name string) (ModelInfo, bool) {
	// Normalize name
	name = strings.TrimSuffix(name, ".bin")
	name = strings.TrimPrefix(name, "ggml-")

	info, ok := ModelCatalog[name]
	return info, ok
}
