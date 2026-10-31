package captions

import (
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

func TestWrapWords(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		maxChars int
		want     []string
	}{
		{
			name:     "short text fits on one line",
			input:    "Hello world",
			maxChars: 42,
			want:     []string{"Hello world"},
		},
		{
			name:     "text wraps to two lines",
			input:    "This is a longer text that should wrap to multiple lines",
			maxChars: 30,
			want:     []string{"This is a longer text that", "should wrap to multiple lines"},
		},
		{
			name:     "empty string returns nil",
			input:    "",
			maxChars: 42,
			want:     nil,
		},
		{
			name:     "whitespace only returns nil",
			input:    "   ",
			maxChars: 42,
			want:     nil,
		},
		{
			name:     "single long word split without loss",
			input:    "supercalifragilisticexpialidocious",
			maxChars: 20,
			want:     []string{"supercalifragilistic", "expialidocious"},
		},
		{
			name:     "preserves overflow lines",
			input:    "one two three four five six seven eight nine ten",
			maxChars: 10,
			want:     []string{"one two", "three four", "five six", "seven", "eight nine", "ten"},
		},
		{
			name:     "zero max characters returns nil",
			input:    "Hello world",
			maxChars: 0,
			want:     nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := wrapWords(tt.input, tt.maxChars)
			if len(got) != len(tt.want) {
				t.Errorf("wrapWords() returned %d lines, want %d lines\ngot: %v\nwant: %v",
					len(got), len(tt.want), got, tt.want)
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("wrapWords() line %d = %q, want %q", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestIsStrongPunct(t *testing.T) {
	tests := []struct {
		word string
		want bool
	}{
		{"hello.", true},
		{"what?", true},
		{"wow!", true},
		{"hello,", false},
		{"hello;", false},
		{"hello:", false},
		{"hello", false},
		{`"hello."`, true},
		{`'test?'`, true},
	}

	for _, tt := range tests {
		t.Run(tt.word, func(t *testing.T) {
			got := isStrongPunct(tt.word)
			if got != tt.want {
				t.Errorf("isStrongPunct(%q) = %v, want %v", tt.word, got, tt.want)
			}
		})
	}
}

func TestIsWeakPunct(t *testing.T) {
	tests := []struct {
		word string
		want bool
	}{
		{"hello,", true},
		{"test;", true},
		{"note:", true},
		{"hello.", false},
		{"what?", false},
		{"hello", false},
	}

	for _, tt := range tests {
		t.Run(tt.word, func(t *testing.T) {
			got := isWeakPunct(tt.word)
			if got != tt.want {
				t.Errorf("isWeakPunct(%q) = %v, want %v", tt.word, got, tt.want)
			}
		})
	}
}

func TestTerminalPunct(t *testing.T) {
	tests := []struct {
		word string
		want byte
	}{
		{"hello.", '.'},
		{"what?", '?'},
		{"wow!", '!'},
		{"hello,", ','},
		{"test;", ';'},
		{"note:", ':'},
		{"hello", 0},
		{"", 0},
		{`"quoted."`, '.'},
	}

	for _, tt := range tests {
		t.Run(tt.word, func(t *testing.T) {
			got := terminalPunct(tt.word)
			if got != tt.want {
				t.Errorf("terminalPunct(%q) = %q, want %q", tt.word, got, tt.want)
			}
		})
	}
}

func TestMergeShortCues(t *testing.T) {
	tests := []struct {
		name   string
		cues   []Cue
		minDur time.Duration
		want   int // expected number of cues after merge
	}{
		{
			name: "no merge needed",
			cues: []Cue{
				{Idx: 1, Start: 0, End: 2 * time.Second, RawText: "First", Lines: []string{"First"}},
				{Idx: 2, Start: 3 * time.Second, End: 5 * time.Second, RawText: "Second", Lines: []string{"Second"}},
			},
			minDur: 600 * time.Millisecond,
			want:   2,
		},
		{
			name: "short cue merged",
			cues: []Cue{
				{Idx: 1, Start: 0, End: 300 * time.Millisecond, RawText: "Hi", Lines: []string{"Hi"}},
				{Idx: 2, Start: 500 * time.Millisecond, End: 2 * time.Second, RawText: "there", Lines: []string{"there"}},
			},
			minDur: 600 * time.Millisecond,
			want:   1,
		},
		{
			name: "single cue unchanged",
			cues: []Cue{
				{Idx: 1, Start: 0, End: 2 * time.Second, RawText: "Only one", Lines: []string{"Only one"}},
			},
			minDur: 600 * time.Millisecond,
			want:   1,
		},
		{
			name:   "empty cues",
			cues:   []Cue{},
			minDur: 600 * time.Millisecond,
			want:   0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mergeShortCues(tt.cues, tt.minDur, 42, 2)
			if len(got) != tt.want {
				t.Errorf("mergeShortCues() returned %d cues, want %d", len(got), tt.want)
			}
		})
	}
}

// mockSegment creates a whisper.Segment for testing
func mockSegment(start, end time.Duration, text string) whisper.Segment {
	return whisper.Segment{
		Start: start,
		End:   end,
		Text:  text,
	}
}

func TestSegmentsToCues(t *testing.T) {
	tests := []struct {
		name     string
		segments []whisper.Segment
		maxChars int
		maxLines int
		wantLen  int
	}{
		{
			name: "basic conversion",
			segments: []whisper.Segment{
				mockSegment(0, 2*time.Second, "Hello world"),
				mockSegment(3*time.Second, 5*time.Second, "Goodbye world"),
			},
			maxChars: 42,
			maxLines: 2,
			wantLen:  2,
		},
		{
			name: "empty segments",
			segments: []whisper.Segment{
				mockSegment(0, 2*time.Second, ""),
			},
			maxChars: 42,
			maxLines: 2,
			wantLen:  0,
		},
		{
			name:     "nil segments",
			segments: nil,
			maxChars: 42,
			maxLines: 2,
			wantLen:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SegmentsToCues(tt.segments, tt.maxChars, tt.maxLines)
			if len(got) != tt.wantLen {
				t.Errorf("SegmentsToCues() returned %d cues, want %d", len(got), tt.wantLen)
			}
		})
	}
}

func TestFindSplitIdx(t *testing.T) {
	tests := []struct {
		name  string
		words []string
		want  int // -1 means no split found
	}{
		{
			name:  "split at period",
			words: []string{"Hello", "world."},
			want:  1,
		},
		{
			name:  "split at comma",
			words: []string{"Hello,", "world"},
			want:  0,
		},
		{
			name:  "no punctuation",
			words: []string{"Hello", "world"},
			want:  -1,
		},
		{
			name:  "prefer strong over weak",
			words: []string{"Hello,", "world."},
			want:  1, // period is stronger
		},
		{
			name:  "empty words",
			words: []string{},
			want:  -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := findSplitIdx(tt.words)
			if got != tt.want {
				t.Errorf("findSplitIdx(%v) = %d, want %d", tt.words, got, tt.want)
			}
		})
	}
}

func TestSegmentsToCuesPreservesTextAndLimits(t *testing.T) {
	tests := []struct {
		name               string
		text               string
		maxChars, maxLines int
	}{
		{"long segment", "one two three four five six seven eight nine ten eleven twelve", 10, 2},
		{"Unicode words", "Café déjà vu Grüße κόσμε привет мир", 10, 2},
		{"unbroken Unicode", "你好世界这是一个没有空格的字幕测试", 4, 2},
		{"long word", "supercalifragilisticexpialidocious", 8, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			start, end := 3*time.Second, 13*time.Second
			cues := SegmentsToCues([]whisper.Segment{mockSegment(start, end, tt.text)}, tt.maxChars, tt.maxLines)
			if len(cues) < 2 {
				t.Fatalf("expected multiple cues, got %d: %+v", len(cues), cues)
			}
			var rendered, raw strings.Builder
			for i, cue := range cues {
				if cue.Idx != i+1 || len(cue.Lines) > tt.maxLines || len(cue.Lines) == 0 {
					t.Errorf("invalid cue: %+v", cue)
				}
				if cue.End <= cue.Start || (i > 0 && cue.Start != cues[i-1].End) {
					t.Errorf("cue timing is not positive and continuous: %+v", cue)
				}
				for _, line := range cue.Lines {
					if !utf8.ValidString(line) || utf8.RuneCountInString(line) > tt.maxChars {
						t.Errorf("invalid or overlong line: %q", line)
					}
					rendered.WriteString(line)
				}
				raw.WriteString(cue.RawText)
			}
			withoutSpaces := func(s string) string { return strings.Join(strings.Fields(s), "") }
			for name, got := range map[string]string{"rendered": rendered.String(), "raw": raw.String()} {
				if withoutSpaces(got) != withoutSpaces(tt.text) {
					t.Errorf("%s text lost or changed: got %q, want %q", name, got, tt.text)
				}
			}
			if cues[0].Start != start || cues[len(cues)-1].End != end {
				t.Error("split cues must cover the original segment interval")
			}
		})
	}
}

func TestSegmentsToCuesMergeRespectsLimitsAndGaps(t *testing.T) {
	for _, gap := range []time.Duration{0, 10 * time.Second} {
		segs := []whisper.Segment{
			mockSegment(0, 300*time.Millisecond, "hello"),
			mockSegment(300*time.Millisecond+gap, 2*time.Second+gap, "world"),
		}
		cues := SegmentsToCues(segs, 5, 1)
		if len(cues) != 2 {
			t.Errorf("gap %v: merging should preserve limits, got %+v", gap, cues)
		}
	}
	cues := SegmentsToCues([]whisper.Segment{
		mockSegment(0, 300*time.Millisecond, "Hi"),
		mockSegment(10*time.Second, 12*time.Second, "there"),
	}, 42, 2)
	if len(cues) != 2 {
		t.Error("short cues separated by silence must not merge")
	}
}
