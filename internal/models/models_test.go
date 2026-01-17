package models

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultModelDir(t *testing.T) {
	dir := DefaultModelDir()
	if dir == "" {
		t.Error("DefaultModelDir should not return empty string")
	}
}

func TestDefaultModelDirXDG(t *testing.T) {
	// Test XDG_DATA_HOME override
	oldVal := os.Getenv("XDG_DATA_HOME")
	defer os.Setenv("XDG_DATA_HOME", oldVal)

	os.Setenv("XDG_DATA_HOME", "/custom/data")
	dir := DefaultModelDir()
	if dir != "/custom/data/opencaption/models" {
		t.Errorf("DefaultModelDir with XDG_DATA_HOME = %q, want %q", dir, "/custom/data/opencaption/models")
	}
}

func TestKnownModels(t *testing.T) {
	// Check that common models are defined
	expectedModels := []string{
		"tiny", "tiny.en",
		"base", "base.en",
		"small", "small.en",
		"medium", "medium.en",
		"large-v1", "large-v2", "large-v3", "large",
		"large-v3-turbo",
	}

	for _, model := range expectedModels {
		if _, ok := KnownModels[model]; !ok {
			t.Errorf("Model %q should be in KnownModels", model)
		}
	}

	// Check that all URLs are valid (start with https)
	for name, url := range KnownModels {
		if len(url) < 8 || url[:8] != "https://" {
			t.Errorf("Model %q has invalid URL: %q", name, url)
		}
	}
}

func TestModelCatalog(t *testing.T) {
	// Check that catalog entries match known models
	for name := range ModelCatalog {
		if _, ok := KnownModels[name]; !ok {
			t.Errorf("ModelCatalog entry %q not in KnownModels", name)
		}
	}
}

func TestGetModelInfo(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"tiny", true},
		{"base.en", true},
		{"large-v3", true},
		{"unknown", false},
		{"ggml-tiny.bin", true}, // Should normalize
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ok := GetModelInfo(tt.name)
			if ok != tt.want {
				t.Errorf("GetModelInfo(%q) ok = %v, want %v", tt.name, ok, tt.want)
			}
		})
	}
}

func TestListKnownModels(t *testing.T) {
	list := ListKnownModels()
	if list == "" {
		t.Error("ListKnownModels should return non-empty string")
	}
	// Should contain some known models
	if !strings.Contains(list, "tiny") {
		t.Errorf("ListKnownModels should contain 'tiny', got: %s", list)
	}
}

func TestResolveModelExistingFile(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a mock model file
	modelPath := filepath.Join(tmpDir, "ggml-test.bin")
	if err := os.WriteFile(modelPath, []byte("mock model"), 0644); err != nil {
		t.Fatalf("Failed to create mock model: %v", err)
	}

	// Resolve by absolute path
	resolved, err := ResolveModel(modelPath, "", false)
	if err != nil {
		t.Fatalf("ResolveModel failed: %v", err)
	}
	if resolved != modelPath {
		t.Errorf("ResolveModel = %q, want %q", resolved, modelPath)
	}
}

func TestResolveModelWithContext(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a mock model file
	modelPath := filepath.Join(tmpDir, "ggml-test.bin")
	if err := os.WriteFile(modelPath, []byte("mock model"), 0644); err != nil {
		t.Fatalf("Failed to create mock model: %v", err)
	}

	ctx := context.Background()
	resolved, err := ResolveModelWithContext(ctx, modelPath, "", false)
	if err != nil {
		t.Fatalf("ResolveModelWithContext failed: %v", err)
	}
	if resolved != modelPath {
		t.Errorf("ResolveModelWithContext = %q, want %q", resolved, modelPath)
	}
}

func TestResolveModelInModelDir(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a mock model file in model directory
	modelPath := filepath.Join(tmpDir, "ggml-base.en.bin")
	if err := os.WriteFile(modelPath, []byte("mock model"), 0644); err != nil {
		t.Fatalf("Failed to create mock model: %v", err)
	}

	// Resolve by name with model directory
	resolved, err := ResolveModel("ggml-base.en.bin", tmpDir, false)
	if err != nil {
		t.Fatalf("ResolveModel failed: %v", err)
	}
	if resolved != modelPath {
		t.Errorf("ResolveModel = %q, want %q", resolved, modelPath)
	}
}

func TestResolveModelWithPrefix(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a mock model file with ggml- prefix
	modelPath := filepath.Join(tmpDir, "ggml-tiny.bin")
	if err := os.WriteFile(modelPath, []byte("mock model"), 0644); err != nil {
		t.Fatalf("Failed to create mock model: %v", err)
	}

	// Resolve by short name (without ggml- prefix)
	resolved, err := ResolveModel("tiny", tmpDir, false)
	if err != nil {
		t.Fatalf("ResolveModel failed: %v", err)
	}
	if resolved != modelPath {
		t.Errorf("ResolveModel = %q, want %q", resolved, modelPath)
	}
}

func TestResolveModelNotFound(t *testing.T) {
	_, err := ResolveModel("nonexistent.bin", "", false)
	if err == nil {
		t.Error("ResolveModel should fail for nonexistent model")
	}
}

func TestResolveModelNotFoundInDir(t *testing.T) {
	tmpDir := t.TempDir()

	_, err := ResolveModel("nonexistent.bin", tmpDir, false)
	if err == nil {
		t.Error("ResolveModel should fail for nonexistent model in directory")
	}
}

func TestListAvailable(t *testing.T) {
	tmpDir := t.TempDir()

	// Create some mock model files
	models := []string{"ggml-tiny.bin", "ggml-base.en.bin", "ggml-small.bin"}
	for _, m := range models {
		path := filepath.Join(tmpDir, m)
		if err := os.WriteFile(path, []byte("mock"), 0644); err != nil {
			t.Fatalf("Failed to create mock model: %v", err)
		}
	}

	// Also create a non-model file
	if err := os.WriteFile(filepath.Join(tmpDir, "readme.txt"), []byte("test"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// List available models
	available, err := ListAvailable(tmpDir)
	if err != nil {
		t.Fatalf("ListAvailable failed: %v", err)
	}

	if len(available) != 3 {
		t.Errorf("ListAvailable found %d models, want 3", len(available))
	}

	// Check that non-model files are excluded
	for _, m := range available {
		if m == "readme.txt" {
			t.Error("ListAvailable should not include non-model files")
		}
	}

	// Check that results are sorted
	for i := 1; i < len(available); i++ {
		if available[i-1] > available[i] {
			t.Error("ListAvailable results should be sorted")
		}
	}
}

func TestListAvailableEmpty(t *testing.T) {
	tmpDir := t.TempDir()

	available, err := ListAvailable(tmpDir)
	if err != nil {
		t.Fatalf("ListAvailable failed: %v", err)
	}

	if len(available) != 0 {
		t.Errorf("ListAvailable should return empty for empty directory, got %d", len(available))
	}
}

func TestListAvailableNonexistent(t *testing.T) {
	available, err := ListAvailable("/nonexistent/path")
	if err != nil {
		t.Fatalf("ListAvailable should not error for nonexistent path: %v", err)
	}
	if available != nil {
		t.Errorf("ListAvailable should return nil for nonexistent path, got %v", available)
	}
}
