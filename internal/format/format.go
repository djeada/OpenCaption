package format

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"opencaption/internal/captions"
)

// JSONCue represents a cue in JSON output format.
type JSONCue struct {
	Index    int      `json:"index"`
	Start    float64  `json:"start"`
	End      float64  `json:"end"`
	StartStr string   `json:"start_formatted"`
	EndStr   string   `json:"end_formatted"`
	Text     string   `json:"text"`
	Lines    []string `json:"lines"`
}

// JSONOutput represents the complete JSON output structure.
type JSONOutput struct {
	Version  string    `json:"version"`
	Format   string    `json:"format"`
	CueCount int       `json:"cue_count"`
	Duration float64   `json:"duration_seconds,omitempty"`
	Cues     []JSONCue `json:"cues"`
}

// WriteJSON writes cues as JSON.
func WriteJSON(w io.Writer, cues []captions.Cue) error {
	output := JSONOutput{
		Version:  "1.0",
		Format:   "opencaption",
		CueCount: len(cues),
		Cues:     make([]JSONCue, len(cues)),
	}

	var maxEnd float64
	for i, c := range cues {
		startSec := c.Start.Seconds()
		endSec := c.End.Seconds()
		if endSec > maxEnd {
			maxEnd = endSec
		}

		output.Cues[i] = JSONCue{
			Index:    c.Idx,
			Start:    startSec,
			End:      endSec,
			StartStr: tsVTT(c.Start),
			EndStr:   tsVTT(c.End),
			Text:     c.RawText,
			Lines:    c.Lines,
		}
	}
	output.Duration = maxEnd

	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	return encoder.Encode(output)
}

func tsVTT(t time.Duration) string {
	if t < 0 {
		t = 0
	}
	h := t / time.Hour
	t -= h * time.Hour
	m := t / time.Minute
	t -= m * time.Minute
	s := t / time.Second
	ms := (t - s*time.Second) / time.Millisecond
	return fmt.Sprintf("%02d:%02d:%02d.%03d", h, m, s, ms)
}

func tsSRT(t time.Duration) string {
	return strings.ReplaceAll(tsVTT(t), ".", ",")
}

// WriteVTT writes cues as a WEBVTT document.
func WriteVTT(w io.Writer, cues []captions.Cue) {
	fmt.Fprintln(w, "WEBVTT\n")
	for _, c := range cues {
		fmt.Fprintf(w, "%s --> %s\n", tsVTT(c.Start), tsVTT(c.End))
		for _, line := range c.Lines {
			fmt.Fprintln(w, line)
		}
		fmt.Fprintln(w)
	}
}

// WriteSRT writes cues in SRT format.
func WriteSRT(w io.Writer, cues []captions.Cue) {
	for _, c := range cues {
		fmt.Fprintln(w, c.Idx)
		fmt.Fprintf(w, "%s --> %s\n", tsSRT(c.Start), tsSRT(c.End))
		for _, line := range c.Lines {
			fmt.Fprintln(w, line)
		}
		fmt.Fprintln(w)
	}
}
