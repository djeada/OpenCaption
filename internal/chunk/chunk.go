package chunk

import (
	"strings"
	"time"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

// Chunk represents a PCM window with a sample offset.
type Chunk struct {
	Start int
	PCM   []float32
}

// ChunkPCM splits PCM into overlapping windows.
func ChunkPCM(pcm []float32, sr, windowSec, overlapSec int) []Chunk {
	win := windowSec * sr
	ovl := overlapSec * sr
	if win <= 0 {
		return []Chunk{{Start: 0, PCM: pcm}}
	}
	var out []Chunk
	for start := 0; start < len(pcm); start += (win - ovl) {
		end := start + win
		if end > len(pcm) {
			end = len(pcm)
		}
		out = append(out, Chunk{Start: start, PCM: pcm[start:end]})
		if end == len(pcm) {
			break
		}
	}
	return out
}

// DedupeOverlap drops near-duplicate segments created by overlap windows.
func DedupeOverlap(segs []whisper.Segment, overlap time.Duration) []whisper.Segment {
	if len(segs) < 2 {
		return segs
	}
	out := []whisper.Segment{segs[0]}
	for i := 1; i < len(segs); i++ {
		prev := out[len(out)-1]
		cur := segs[i]
		// if same (or near-same) text within small time gap, drop duplicate
		if absDuration(prev.Start-cur.Start) < overlap+250*time.Millisecond {
			if nearDuplicate(prev.Text, cur.Text) {
				continue
			}
		}
		out = append(out, cur)
	}
	return out
}

func absDuration(d time.Duration) time.Duration {
	if d < 0 {
		return -d
	}
	return d
}

func nearDuplicate(a, b string) bool {
	na := normalizeText(a)
	nb := normalizeText(b)
	if na == "" || nb == "" {
		return false
	}
	if na == nb {
		return true
	}
	if strings.HasPrefix(na, nb) || strings.HasPrefix(nb, na) {
		return true
	}
	return tokenSimilarity(na, nb) >= 0.7
}

func normalizeText(s string) string {
	words := strings.Fields(strings.ToLower(s))
	out := make([]string, 0, len(words))
	for _, w := range words {
		w = strings.Trim(w, ".,!?;:\"'()[]{}")
		if w == "" {
			continue
		}
		out = append(out, w)
	}
	return strings.Join(out, " ")
}

func tokenSimilarity(a, b string) float64 {
	aw := strings.Fields(a)
	bw := strings.Fields(b)
	if len(aw) == 0 || len(bw) == 0 {
		return 0
	}
	set := make(map[string]struct{}, len(aw))
	for _, w := range aw {
		set[w] = struct{}{}
	}
	overlap := 0
	for _, w := range bw {
		if _, ok := set[w]; ok {
			overlap++
		}
	}
	maxLen := len(aw)
	if len(bw) > maxLen {
		maxLen = len(bw)
	}
	return float64(overlap) / float64(maxLen)
}
