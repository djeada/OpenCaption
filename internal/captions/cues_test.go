package captions

import (
	"testing"
	"time"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

func TestWrapWords(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		maxChars int
		maxLines int
		want     []string
	}{
		{
			name:     "short text fits on one line",
			input:    "Hello world",
			maxChars: 42,
			maxLines: 2,
			want:     []string{"Hello world"},
		},
		{
			name:     "text wraps to two lines",
			input:    "This is a longer text that should wrap to multiple lines",
			maxChars: 30,
			maxLines: 2,
			want:     []string{"This is a longer text that", "should wrap to multiple lines"},
		},
		{
			name:     "empty string returns nil",
			input:    "",
			maxChars: 42,
			maxLines: 2,
			want:     nil,
		},
		{
			name:     "whitespace only returns nil",
			input:    "   ",
			maxChars: 42,
			maxLines: 2,
			want:     nil,
		},
		{
			name:     "single long word truncated",
			input:    "supercalifragilisticexpialidocious",
			maxChars: 20,
			maxLines: 1,
			want:     []string{"supercalifragilistic"},
		},
		{
			name:     "respects max lines",
			input:    "one two three four five six seven eight nine ten",
			maxChars: 10,
			maxLines: 2,
			want:     []string{"one two", "three four"},
		},
		{
			name:     "zero max lines returns nil",
			input:    "Hello world",
			maxChars: 42,
			maxLines: 0,
			want:     nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := wrapWords(tt.input, tt.maxChars, tt.maxLines)
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

func TestSoftTruncate(t *testing.T) {
	tests := []struct {
		name  string
		input string
		max   int
		want  string
	}{
		{"short string unchanged", "Hello", 10, "Hello"},
		{"exact length unchanged", "Hello", 5, "Hello"},
		{"truncate at space", "Hello world foo", 12, "Hello world"},
		{"no space truncate at max", "Helloworld", 5, "Hello"},
		{"empty string", "", 10, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := softTruncate(tt.input, tt.max)
			if got != tt.want {
				t.Errorf("softTruncate(%q, %d) = %q, want %q", tt.input, tt.max, got, tt.want)
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
			got := mergeShortCues(tt.cues, tt.minDur)
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
