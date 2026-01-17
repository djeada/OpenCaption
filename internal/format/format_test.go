package format

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"opencaption/internal/captions"
)

func TestTsVTT(t *testing.T) {
	tests := []struct {
		name     string
		duration time.Duration
		want     string
	}{
		{"zero", 0, "00:00:00.000"},
		{"negative", -1 * time.Second, "00:00:00.000"},
		{"one second", 1 * time.Second, "00:00:01.000"},
		{"one minute", 1 * time.Minute, "00:01:00.000"},
		{"one hour", 1 * time.Hour, "01:00:00.000"},
		{"mixed", 1*time.Hour + 23*time.Minute + 45*time.Second + 678*time.Millisecond, "01:23:45.678"},
		{"large", 99*time.Hour + 59*time.Minute + 59*time.Second + 999*time.Millisecond, "99:59:59.999"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tsVTT(tt.duration)
			if got != tt.want {
				t.Errorf("tsVTT(%v) = %q, want %q", tt.duration, got, tt.want)
			}
		})
	}
}

func TestTsSRT(t *testing.T) {
	tests := []struct {
		name     string
		duration time.Duration
		want     string
	}{
		{"zero", 0, "00:00:00,000"},
		{"mixed", 1*time.Hour + 23*time.Minute + 45*time.Second + 678*time.Millisecond, "01:23:45,678"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tsSRT(tt.duration)
			if got != tt.want {
				t.Errorf("tsSRT(%v) = %q, want %q", tt.duration, got, tt.want)
			}
		})
	}
}

func TestWriteVTT(t *testing.T) {
	cues := []captions.Cue{
		{
			Idx:   1,
			Start: 0,
			End:   2 * time.Second,
			Lines: []string{"Hello, world!"},
		},
		{
			Idx:   2,
			Start: 3 * time.Second,
			End:   5 * time.Second,
			Lines: []string{"This is a test.", "Multiple lines."},
		},
	}

	var buf bytes.Buffer
	WriteVTT(&buf, cues)
	output := buf.String()

	// Check header
	if !strings.HasPrefix(output, "WEBVTT\n") {
		t.Errorf("VTT output should start with 'WEBVTT\\n', got: %q", output[:min(20, len(output))])
	}

	// Check timestamps
	if !strings.Contains(output, "00:00:00.000 --> 00:00:02.000") {
		t.Error("VTT output missing first cue timestamps")
	}
	if !strings.Contains(output, "00:00:03.000 --> 00:00:05.000") {
		t.Error("VTT output missing second cue timestamps")
	}

	// Check content
	if !strings.Contains(output, "Hello, world!") {
		t.Error("VTT output missing first cue text")
	}
	if !strings.Contains(output, "This is a test.") {
		t.Error("VTT output missing second cue first line")
	}
	if !strings.Contains(output, "Multiple lines.") {
		t.Error("VTT output missing second cue second line")
	}
}

func TestWriteSRT(t *testing.T) {
	cues := []captions.Cue{
		{
			Idx:   1,
			Start: 0,
			End:   2 * time.Second,
			Lines: []string{"Hello, world!"},
		},
		{
			Idx:   2,
			Start: 3 * time.Second,
			End:   5*time.Second + 500*time.Millisecond,
			Lines: []string{"Another line"},
		},
	}

	var buf bytes.Buffer
	WriteSRT(&buf, cues)
	output := buf.String()

	// Check index
	if !strings.Contains(output, "1\n") {
		t.Error("SRT output missing cue index 1")
	}
	if !strings.Contains(output, "2\n") {
		t.Error("SRT output missing cue index 2")
	}

	// Check SRT timestamp format (comma instead of period)
	if !strings.Contains(output, "00:00:00,000 --> 00:00:02,000") {
		t.Error("SRT output missing first cue timestamps with comma separator")
	}
	if !strings.Contains(output, "00:00:03,000 --> 00:00:05,500") {
		t.Error("SRT output missing second cue timestamps")
	}
}

func TestWriteJSON(t *testing.T) {
	cues := []captions.Cue{
		{
			Idx:     1,
			Start:   0,
			End:     2 * time.Second,
			Lines:   []string{"Hello, world!"},
			RawText: "Hello, world!",
		},
		{
			Idx:     2,
			Start:   3 * time.Second,
			End:     5 * time.Second,
			Lines:   []string{"Line 1", "Line 2"},
			RawText: "Line 1 Line 2",
		},
	}

	var buf bytes.Buffer
	err := WriteJSON(&buf, cues)
	if err != nil {
		t.Fatalf("WriteJSON failed: %v", err)
	}

	// Parse the output
	var output JSONOutput
	if err := json.Unmarshal(buf.Bytes(), &output); err != nil {
		t.Fatalf("Failed to parse JSON output: %v", err)
	}

	// Check structure
	if output.Version != "1.0" {
		t.Errorf("Version = %q, want %q", output.Version, "1.0")
	}
	if output.CueCount != 2 {
		t.Errorf("CueCount = %d, want %d", output.CueCount, 2)
	}
	if output.Duration != 5.0 {
		t.Errorf("Duration = %f, want %f", output.Duration, 5.0)
	}
	if len(output.Cues) != 2 {
		t.Fatalf("len(Cues) = %d, want %d", len(output.Cues), 2)
	}

	// Check first cue
	if output.Cues[0].Index != 1 {
		t.Errorf("Cues[0].Index = %d, want %d", output.Cues[0].Index, 1)
	}
	if output.Cues[0].Start != 0.0 {
		t.Errorf("Cues[0].Start = %f, want %f", output.Cues[0].Start, 0.0)
	}
	if output.Cues[0].End != 2.0 {
		t.Errorf("Cues[0].End = %f, want %f", output.Cues[0].End, 2.0)
	}
	if output.Cues[0].Text != "Hello, world!" {
		t.Errorf("Cues[0].Text = %q, want %q", output.Cues[0].Text, "Hello, world!")
	}

	// Check second cue has multiple lines
	if len(output.Cues[1].Lines) != 2 {
		t.Errorf("len(Cues[1].Lines) = %d, want %d", len(output.Cues[1].Lines), 2)
	}
}
