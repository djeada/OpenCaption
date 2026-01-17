package chunk

import (
	"testing"
	"time"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

func TestChunkPCM(t *testing.T) {
	tests := []struct {
		name       string
		pcmLen     int
		sr         int
		windowSec  int
		overlapSec int
		wantLen    int
	}{
		{
			name:       "short audio no chunking",
			pcmLen:     16000, // 1 second
			sr:         16000,
			windowSec:  0,
			overlapSec: 0,
			wantLen:    1,
		},
		{
			name:       "exact one window",
			pcmLen:     16000 * 60, // 60 seconds
			sr:         16000,
			windowSec:  60,
			overlapSec: 0,
			wantLen:    1,
		},
		{
			name:       "two windows with overlap",
			pcmLen:     16000 * 90, // 90 seconds
			sr:         16000,
			windowSec:  60,
			overlapSec: 10,
			wantLen:    2,
		},
		{
			name:       "multiple chunks",
			pcmLen:     16000 * 180, // 3 minutes
			sr:         16000,
			windowSec:  60,
			overlapSec: 1,
			wantLen:    4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pcm := make([]float32, tt.pcmLen)
			chunks := ChunkPCM(pcm, tt.sr, tt.windowSec, tt.overlapSec)
			if len(chunks) != tt.wantLen {
				t.Errorf("ChunkPCM returned %d chunks, want %d", len(chunks), tt.wantLen)
			}
		})
	}
}

func TestChunkPCMStartPositions(t *testing.T) {
	sr := 16000
	windowSec := 60
	overlapSec := 10
	pcm := make([]float32, sr*120) // 2 minutes

	chunks := ChunkPCM(pcm, sr, windowSec, overlapSec)

	// First chunk should start at 0
	if chunks[0].Start != 0 {
		t.Errorf("First chunk Start = %d, want 0", chunks[0].Start)
	}

	// Second chunk should start at window - overlap
	expectedStart := (windowSec - overlapSec) * sr
	if len(chunks) > 1 && chunks[1].Start != expectedStart {
		t.Errorf("Second chunk Start = %d, want %d", chunks[1].Start, expectedStart)
	}
}

func TestAbsDuration(t *testing.T) {
	tests := []struct {
		input time.Duration
		want  time.Duration
	}{
		{0, 0},
		{time.Second, time.Second},
		{-time.Second, time.Second},
		{-5 * time.Minute, 5 * time.Minute},
	}

	for _, tt := range tests {
		got := absDuration(tt.input)
		if got != tt.want {
			t.Errorf("absDuration(%v) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

func TestNormalizeText(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"Hello, World!", "hello world"},
		{"  Multiple   Spaces  ", "multiple spaces"},
		{"Punctuation...", "punctuation"},
		{`"Quoted text"`, "quoted text"},
		{"", ""},
		{"UPPERCASE", "uppercase"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := normalizeText(tt.input)
			if got != tt.want {
				t.Errorf("normalizeText(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestTokenSimilarity(t *testing.T) {
	tests := []struct {
		a, b   string
		minSim float64
		maxSim float64
	}{
		{"hello world", "hello world", 1.0, 1.0},
		{"hello world", "hello there", 0.4, 0.6},
		{"completely different", "nothing similar here", 0.0, 0.3},
		{"", "hello", 0.0, 0.0},
		{"hello", "", 0.0, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.a+"_"+tt.b, func(t *testing.T) {
			sim := tokenSimilarity(tt.a, tt.b)
			if sim < tt.minSim || sim > tt.maxSim {
				t.Errorf("tokenSimilarity(%q, %q) = %f, want between %f and %f",
					tt.a, tt.b, sim, tt.minSim, tt.maxSim)
			}
		})
	}
}

func TestNearDuplicate(t *testing.T) {
	tests := []struct {
		a, b string
		want bool
	}{
		{"Hello, world!", "Hello, world!", true},
		{"Hello world", "hello world", true},
		{"The quick brown fox", "The quick brown", true}, // prefix match
		{"Hello", "Goodbye", false},
		{"", "Hello", false},
		{"Hello", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.a+"_"+tt.b, func(t *testing.T) {
			got := nearDuplicate(tt.a, tt.b)
			if got != tt.want {
				t.Errorf("nearDuplicate(%q, %q) = %v, want %v", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

func mockSegment(start, end time.Duration, text string) whisper.Segment {
	return whisper.Segment{
		Start: start,
		End:   end,
		Text:  text,
	}
}

func TestDedupeOverlap(t *testing.T) {
	tests := []struct {
		name    string
		segs    []whisper.Segment
		overlap time.Duration
		wantLen int
	}{
		{
			name:    "empty segments",
			segs:    nil,
			overlap: time.Second,
			wantLen: 0,
		},
		{
			name: "single segment",
			segs: []whisper.Segment{
				mockSegment(0, 2*time.Second, "Hello"),
			},
			overlap: time.Second,
			wantLen: 1,
		},
		{
			name: "no duplicates",
			segs: []whisper.Segment{
				mockSegment(0, 2*time.Second, "Hello"),
				mockSegment(5*time.Second, 7*time.Second, "World"),
			},
			overlap: time.Second,
			wantLen: 2,
		},
		{
			name: "duplicate removed",
			segs: []whisper.Segment{
				mockSegment(0, 2*time.Second, "Hello world"),
				mockSegment(100*time.Millisecond, 2*time.Second, "Hello world"),
			},
			overlap: time.Second,
			wantLen: 1,
		},
		{
			name: "near duplicate removed",
			segs: []whisper.Segment{
				mockSegment(0, 2*time.Second, "Hello, world!"),
				mockSegment(200*time.Millisecond, 2*time.Second, "Hello world"),
			},
			overlap: time.Second,
			wantLen: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DedupeOverlap(tt.segs, tt.overlap)
			if len(got) != tt.wantLen {
				t.Errorf("DedupeOverlap returned %d segments, want %d", len(got), tt.wantLen)
			}
		})
	}
}
