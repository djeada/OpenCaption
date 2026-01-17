package models

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
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

// DefaultModelDir returns the default model directory path.
func DefaultModelDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "./models"
	}
	return filepath.Join(home, ".local", "share", "opencaption", "models")
}

// ResolveModel resolves a model name or path to an actual file path.
// If modelDir is provided and the model doesn't exist, it will attempt to download it.
func ResolveModel(model, modelDir string, autoDownload bool) (string, error) {
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
			return downloadModel(model, modelDir)
		}
	}

	return "", fmt.Errorf("model not found: %s (use --model-dir or provide full path)", model)
}

// downloadModel downloads a model to the specified directory.
func downloadModel(model, modelDir string) (string, error) {
	// Normalize model name
	name := strings.TrimSuffix(filepath.Base(model), ".bin")
	name = strings.TrimPrefix(name, "ggml-")

	url, ok := KnownModels[name]
	if !ok {
		return "", fmt.Errorf("unknown model: %s (known: %s)", name, listKnownModels())
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

	fmt.Fprintf(os.Stderr, "Downloading model %s...\n", name)

	// Download with progress
	resp, err := http.Get(url)
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

	// Copy with progress reporting
	written, err := io.Copy(out, &progressReader{r: resp.Body, total: resp.ContentLength})
	out.Close()
	if err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("download model: %w", err)
	}

	// Rename to final path
	if err := os.Rename(tmpPath, destPath); err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("rename model file: %w", err)
	}

	fmt.Fprintf(os.Stderr, "\nDownloaded %s (%.1f MB)\n", destName, float64(written)/1024/1024)
	return destPath, nil
}

type progressReader struct {
	r       io.Reader
	total   int64
	current int64
	lastPct int
}

func (pr *progressReader) Read(p []byte) (int, error) {
	n, err := pr.r.Read(p)
	pr.current += int64(n)

	if pr.total > 0 {
		pct := int(100 * pr.current / pr.total)
		if pct != pr.lastPct && pct%5 == 0 {
			fmt.Fprintf(os.Stderr, "\rDownloading... %d%%", pct)
			pr.lastPct = pct
		}
	}
	return n, err
}

func listKnownModels() string {
	var names []string
	for k := range KnownModels {
		names = append(names, k)
	}
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
	return models, nil
}
