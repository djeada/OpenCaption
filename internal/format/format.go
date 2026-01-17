package format

import (
	"fmt"
	"io"
	"strings"
	"time"

	"opencaption/internal/captions"
)

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
