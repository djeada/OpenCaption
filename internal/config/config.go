package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// Config holds all configurable options for opencaption.
type Config struct {
	// Input/Output
	Input  string `json:"input" yaml:"input"`
	Output string `json:"output" yaml:"output"`

	// Model
	Model    string `json:"model" yaml:"model"`
	ModelDir string `json:"model_dir" yaml:"model_dir"`

	// FFmpeg
	FFmpegPath string `json:"ffmpeg_path" yaml:"ffmpeg_path"`

	// Device
	Device string `json:"device" yaml:"device"`

	// Processing
	VAD       bool   `json:"vad" yaml:"vad"`
	Language  string `json:"language" yaml:"language"`
	Window    int    `json:"window" yaml:"window"`
	Overlap   int    `json:"overlap" yaml:"overlap"`
	MaxChars  int    `json:"max_chars" yaml:"max_chars"`
	MaxLines  int    `json:"max_lines" yaml:"max_lines"`
	Format    string `json:"format" yaml:"format"`
	Threads   int    `json:"threads" yaml:"threads"`
	BatchMode bool   `json:"batch" yaml:"batch"`
	Recursive bool   `json:"recursive" yaml:"recursive"`
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() *Config {
	return &Config{
		Output:     "captions.vtt",
		Model:      "whisper.cpp/models/ggml-base.en.bin",
		FFmpegPath: "ffmpeg",
		Device:     "auto",
		VAD:        true,
		Window:     60,
		Overlap:    1,
		MaxChars:   42,
		MaxLines:   2,
		Format:     "vtt",
	}
}

// LoadFile loads a config from a JSON or YAML file.
func LoadFile(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	cfg := DefaultConfig()
	ext := strings.ToLower(filepath.Ext(path))

	switch ext {
	case ".json":
		if err := json.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("parse JSON config: %w", err)
		}
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("parse YAML config: %w", err)
		}
	default:
		return nil, errors.New("config file must be .json, .yaml, or .yml")
	}

	return cfg, nil
}

// Presets defines common configuration presets.
var Presets = map[string]*Config{
	"fast": {
		Model:    "ggml-tiny.en.bin",
		Window:   30,
		Overlap:  1,
		VAD:      true,
		MaxChars: 42,
		MaxLines: 2,
		Format:   "vtt",
	},
	"accurate": {
		Model:    "ggml-large-v3.bin",
		Window:   60,
		Overlap:  2,
		VAD:      true,
		MaxChars: 42,
		MaxLines: 2,
		Format:   "vtt",
	},
	"subtitle": {
		Window:   60,
		Overlap:  1,
		VAD:      true,
		MaxChars: 35,
		MaxLines: 2,
		Format:   "srt",
	},
}

// ApplyPreset merges a preset into the config.
func (c *Config) ApplyPreset(name string) error {
	preset, ok := Presets[name]
	if !ok {
		return fmt.Errorf("unknown preset: %s (available: fast, accurate, subtitle)", name)
	}

	// Only apply non-zero preset values
	if preset.Model != "" {
		c.Model = preset.Model
	}
	if preset.Window > 0 {
		c.Window = preset.Window
	}
	if preset.Overlap > 0 {
		c.Overlap = preset.Overlap
	}
	if preset.MaxChars > 0 {
		c.MaxChars = preset.MaxChars
	}
	if preset.MaxLines > 0 {
		c.MaxLines = preset.MaxLines
	}
	if preset.Format != "" {
		c.Format = preset.Format
	}
	c.VAD = preset.VAD

	return nil
}
