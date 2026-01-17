package captions

import (
	"strings"
	"time"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

type Cue struct {
	Idx     int
	Start   time.Duration
	End     time.Duration
	Lines   []string
	RawText string
}

// SegmentsToCues converts whisper segments into wrapped caption cues.
func SegmentsToCues(segs []whisper.Segment, maxChars, maxLines int) []Cue {
	var cues []Cue
	idx := 1
	for _, s := range segs {
		// basic word-safe wrap
		lines := wrapWords(s.Text, maxChars, maxLines)
		if len(lines) == 0 {
			continue
		}
		cues = append(cues, Cue{
			Idx:     idx,
			Start:   s.Start,
			End:     s.End,
			Lines:   lines,
			RawText: s.Text,
		})
		idx++
	}
	// Merge extremely short cues with neighbors (optional polish)
	cues = mergeShortCues(cues, 600*time.Millisecond) // merge cues < 600 ms into neighbors when safe
	return cues
}

func wrapWords(s string, maxChars, maxLines int) []string {
	if maxLines < 1 {
		return nil
	}
	words := strings.Fields(s)
	if len(words) == 0 {
		return nil
	}
	var lines []string
	var curr []string
	for i := 0; i < len(words); i++ {
		w := words[i]
		next := append(curr, w)
		if len(strings.Join(next, " ")) <= maxChars {
			curr = next
			continue
		}
		if len(curr) == 0 {
			lines = append(lines, softTruncate(w, maxChars))
			continue
		}
		if len(lines) == maxLines-1 {
			rest := strings.Join(append(curr, words[i:]...), " ")
			lines = append(lines, softTruncate(rest, maxChars))
			return lines
		}
		if split := findSplitIdx(curr); split >= 0 {
			lines = append(lines, strings.Join(curr[:split+1], " "))
			curr = append([]string{}, curr[split+1:]...)
			curr = append(curr, w)
		} else {
			lines = append(lines, strings.Join(curr, " "))
			curr = []string{w}
		}
	}
	if len(curr) > 0 && len(lines) < maxLines {
		lines = append(lines, strings.Join(curr, " "))
	}
	if len(lines) > maxLines {
		lines = lines[:maxLines]
	}
	return lines
}

func softTruncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	// find last space before max
	if idx := strings.LastIndex(s[:max], " "); idx > 0 {
		return s[:idx]
	}
	return s[:max] // worst case, but still no mid-word split if no spaces exist
}

func mergeShortCues(cues []Cue, minDur time.Duration) []Cue {
	if len(cues) < 2 {
		return cues
	}
	var out []Cue
	for i := 0; i < len(cues); i++ {
		c := cues[i]
		dur := c.End - c.Start
		if dur >= minDur || i == len(cues)-1 {
			out = append(out, c)
			continue
		}
		// merge into next if combined length still readable
		next := cues[i+1]
		merged := Cue{
			Idx:     c.Idx,
			Start:   c.Start,
			End:     next.End,
			Lines:   wrapWords(strings.TrimSpace(c.RawText+" "+next.RawText), 42, 2),
			RawText: strings.TrimSpace(c.RawText + " " + next.RawText),
		}
		out = append(out, merged)
		i++ // skip next
	}
	// reindex
	for i := range out {
		out[i].Idx = i + 1
	}
	return out
}

func findSplitIdx(words []string) int {
	const window = 4
	start := len(words) - window
	if start < 0 {
		start = 0
	}
	for i := len(words) - 1; i >= start; i-- {
		if isStrongPunct(words[i]) {
			return i
		}
	}
	for i := len(words) - 1; i >= start; i-- {
		if isWeakPunct(words[i]) {
			return i
		}
	}
	return -1
}

func isStrongPunct(word string) bool {
	switch terminalPunct(word) {
	case '.', '?', '!':
		return true
	default:
		return false
	}
}

func isWeakPunct(word string) bool {
	switch terminalPunct(word) {
	case ',', ';', ':':
		return true
	default:
		return false
	}
}

func terminalPunct(word string) byte {
	trimmed := strings.TrimRight(word, `"'")]} `)
	if len(trimmed) == 0 {
		return 0
	}
	last := trimmed[len(trimmed)-1]
	switch last {
	case '.', '?', '!', ',', ';', ':':
		return last
	default:
		return 0
	}
}
