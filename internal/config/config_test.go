package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Output != "captions.vtt" {
		t.Errorf("Default Output = %q, want %q", cfg.Output, "captions.vtt")
	}
	if cfg.Format != "vtt" {
		t.Errorf("Default Format = %q, want %q", cfg.Format, "vtt")
	}
	if cfg.VAD != true {
		t.Errorf("Default VAD = %v, want %v", cfg.VAD, true)
	}
	if cfg.Window != 60 {
		t.Errorf("Default Window = %d, want %d", cfg.Window, 60)
	}
	if cfg.Overlap != 1 {
		t.Errorf("Default Overlap = %d, want %d", cfg.Overlap, 1)
	}
	if cfg.MaxChars != 42 {
		t.Errorf("Default MaxChars = %d, want %d", cfg.MaxChars, 42)
	}
	if cfg.MaxLines != 2 {
		t.Errorf("Default MaxLines = %d, want %d", cfg.MaxLines, 2)
	}
	if cfg.Device != "auto" {
		t.Errorf("Default Device = %q, want %q", cfg.Device, "auto")
	}
}

func TestLoadFileJSON(t *testing.T) {
	// Create a temporary JSON config file
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")

	content := `{
		"input": "test.mp4",
		"output": "output.srt",
		"format": "srt",
		"window": 30,
		"max_chars": 35
	}`

	if err := os.WriteFile(configPath, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	cfg, err := LoadFile(configPath)
	if err != nil {
		t.Fatalf("LoadFile failed: %v", err)
	}

	if cfg.Input != "test.mp4" {
		t.Errorf("Input = %q, want %q", cfg.Input, "test.mp4")
	}
	if cfg.Output != "output.srt" {
		t.Errorf("Output = %q, want %q", cfg.Output, "output.srt")
	}
	if cfg.Format != "srt" {
		t.Errorf("Format = %q, want %q", cfg.Format, "srt")
	}
	if cfg.Window != 30 {
		t.Errorf("Window = %d, want %d", cfg.Window, 30)
	}
	if cfg.MaxChars != 35 {
		t.Errorf("MaxChars = %d, want %d", cfg.MaxChars, 35)
	}
	// Check defaults are preserved for unset values
	if cfg.MaxLines != 2 {
		t.Errorf("MaxLines = %d, want default %d", cfg.MaxLines, 2)
	}
}

func TestLoadFileYAML(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	content := `
input: video.mkv
output: captions.json
format: json
vad: false
language: en
window: 45
`

	if err := os.WriteFile(configPath, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	cfg, err := LoadFile(configPath)
	if err != nil {
		t.Fatalf("LoadFile failed: %v", err)
	}

	if cfg.Input != "video.mkv" {
		t.Errorf("Input = %q, want %q", cfg.Input, "video.mkv")
	}
	if cfg.Output != "captions.json" {
		t.Errorf("Output = %q, want %q", cfg.Output, "captions.json")
	}
	if cfg.Format != "json" {
		t.Errorf("Format = %q, want %q", cfg.Format, "json")
	}
	if cfg.VAD != false {
		t.Errorf("VAD = %v, want %v", cfg.VAD, false)
	}
	if cfg.Language != "en" {
		t.Errorf("Language = %q, want %q", cfg.Language, "en")
	}
	if cfg.Window != 45 {
		t.Errorf("Window = %d, want %d", cfg.Window, 45)
	}
}

func TestLoadFileInvalidExtension(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.txt")

	if err := os.WriteFile(configPath, []byte("test"), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	_, err := LoadFile(configPath)
	if err == nil {
		t.Error("LoadFile should fail for .txt extension")
	}
}

func TestLoadFileNotFound(t *testing.T) {
	_, err := LoadFile("/nonexistent/path/config.json")
	if err == nil {
		t.Error("LoadFile should fail for nonexistent file")
	}
}

func TestApplyPreset(t *testing.T) {
	tests := []struct {
		name       string
		preset     string
		wantFormat string
		wantErr    bool
	}{
		{"fast preset", "fast", "vtt", false},
		{"accurate preset", "accurate", "vtt", false},
		{"subtitle preset", "subtitle", "srt", false},
		{"unknown preset", "unknown", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			err := cfg.ApplyPreset(tt.preset)

			if tt.wantErr {
				if err == nil {
					t.Error("ApplyPreset should have failed")
				}
				return
			}

			if err != nil {
				t.Fatalf("ApplyPreset failed: %v", err)
			}

			if cfg.Format != tt.wantFormat {
				t.Errorf("Format = %q, want %q", cfg.Format, tt.wantFormat)
			}
		})
	}
}

func TestApplyPresetFast(t *testing.T) {
	cfg := DefaultConfig()
	if err := cfg.ApplyPreset("fast"); err != nil {
		t.Fatalf("ApplyPreset failed: %v", err)
	}

	if cfg.Window != 30 {
		t.Errorf("Window = %d, want %d", cfg.Window, 30)
	}
	if cfg.Model != "ggml-tiny.en.bin" {
		t.Errorf("Model = %q, want %q", cfg.Model, "ggml-tiny.en.bin")
	}
}

func TestApplyPresetAccurate(t *testing.T) {
	cfg := DefaultConfig()
	if err := cfg.ApplyPreset("accurate"); err != nil {
		t.Fatalf("ApplyPreset failed: %v", err)
	}

	if cfg.Window != 60 {
		t.Errorf("Window = %d, want %d", cfg.Window, 60)
	}
	if cfg.Overlap != 2 {
		t.Errorf("Overlap = %d, want %d", cfg.Overlap, 2)
	}
	if cfg.Model != "ggml-large-v3.bin" {
		t.Errorf("Model = %q, want %q", cfg.Model, "ggml-large-v3.bin")
	}
}

func TestApplyPresetSubtitle(t *testing.T) {
	cfg := DefaultConfig()
	if err := cfg.ApplyPreset("subtitle"); err != nil {
		t.Fatalf("ApplyPreset failed: %v", err)
	}

	if cfg.MaxChars != 35 {
		t.Errorf("MaxChars = %d, want %d", cfg.MaxChars, 35)
	}
	if cfg.Format != "srt" {
		t.Errorf("Format = %q, want %q", cfg.Format, "srt")
	}
}
