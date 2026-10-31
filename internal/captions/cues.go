package captions

import (
	"strings"
	"time"
	"unicode/utf8"

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
// Text that exceeds the line limit is split into additional cues, with timing
// distributed across the original segment in proportion to the text length.
func SegmentsToCues(segs []whisper.Segment, maxChars, maxLines int) []Cue {
	if maxChars < 1 || maxLines < 1 {
		return nil
	}
	var cues []Cue
	for _, s := range segs {
		lines := wrapWords(s.Text, maxChars)
		if len(lines) == 0 {
			continue
		}
		cues = append(cues, Cue{
			Start: s.Start, End: s.End, Lines: lines, RawText: s.Text,
		})
	}

	// Merge before splitting so fragments of an unbroken word are never joined
	// with an invented space. Only merge when the result fits the user's limits.
	cues = mergeShortCues(cues, 600*time.Millisecond, maxChars, maxLines)
	var out []Cue
	for _, c := range cues {
		if len(c.Lines) <= maxLines {
			out = append(out, c)
			continue
		}
		text := []rune(strings.Join(strings.Fields(c.RawText), " "))
		pos := 0
		start := c.Start
		for i := 0; i < len(c.Lines); i += maxLines {
			lines := c.Lines[i:min(i+maxLines, len(c.Lines))]
			rawStart := pos
			for _, line := range lines {
				pos += utf8.RuneCountInString(line)
				for pos < len(text) && text[pos] == ' ' {
					pos++
				}
			}
			end := c.End
			if pos < len(text) {
				end = c.Start + time.Duration(float64(c.End-c.Start)*float64(pos)/float64(len(text)))
			}
			out = append(out, Cue{
				Start: start, End: end, Lines: lines,
				RawText: strings.TrimSpace(string(text[rawStart:pos])),
			})
			start = end
		}
	}
	for i := range out {
		out[i].Idx = i + 1
	}
	return out
}

// wrapWords wraps all text without discarding overflow. Long tokens (including
// languages without spaces) are split at rune boundaries rather than bytes.
func wrapWords(s string, maxChars int) []string {
	if maxChars < 1 {
		return nil
	}
	text := []rune(strings.Join(strings.Fields(s), " "))
	var lines []string
	for start := 0; start < len(text); {
		end := start + min(maxChars, len(text)-start)
		if end < len(text) && text[end] != ' ' {
			for i := end - 1; i > start; i-- {
				if text[i] == ' ' {
					end = i
					break
				}
			}
		}
		line := string(text[start:end])
		if end < len(text) {
			words := strings.Fields(line)
			if split := findSplitIdx(words); split >= 0 && split < len(words)-1 {
				line = strings.Join(words[:split+1], " ")
				end = start + utf8.RuneCountInString(line)
			}
		}
		lines = append(lines, line)
		start = end
		for start < len(text) && text[start] == ' ' {
			start++
		}
	}
	return lines
}

func mergeShortCues(cues []Cue, minDur time.Duration, maxChars, maxLines int) []Cue {
	var out []Cue
	for i := 0; i < len(cues); i++ {
		c := cues[i]
		if c.End-c.Start < minDur && i+1 < len(cues) {
			next := cues[i+1]
			text := strings.TrimSpace(c.RawText + " " + next.RawText)
			lines := wrapWords(text, maxChars)
			if next.Start >= c.Start && next.Start-c.End <= minDur && next.End >= c.End && len(lines) <= maxLines {
				c.End = next.End
				c.RawText = text
				c.Lines = lines
				i++
			}
		}
		out = append(out, c)
	}
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
