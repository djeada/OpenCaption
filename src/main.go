package main

import (
	"bufio"
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

// ---------- CLI flags ----------
var (
	inPath      = flag.String("in", "", "Input audio/video file")
	outPath     = flag.String("out", "captions.vtt", "Output captions file (.vtt or .srt)")
	modelPath   = flag.String("model", "whisper.cpp/models/ggml-base.en.bin", "Path to ggml/gguf model")
	lang        = flag.String("lang", "", "Language (e.g. 'en'); empty = auto")
	windowSec   = flag.Int("window", 60, "Chunk window seconds (0 = whole file)")
	overlapSec  = flag.Int("overlap", 1, "Chunk overlap seconds (only if window > 0)")
	maxChars    = flag.Int("maxchars", 42, "Max characters per line")
	maxLines    = flag.Int("maxlines", 2, "Max lines per cue")
	format      = flag.String("format", "vtt", "Caption format: vtt | srt (default vtt)")
	threads     = flag.Int("threads", 0, "Threads (0 = auto)")
)

func fail(err error) {
	fmt.Fprintln(os.Stderr, "Error:", err)
	os.Exit(1)
}

func main() {
	flag.Parse()
	if *inPath == "" {
		fail(errors.New("please provide -in <file>"))
	}
	if !strings.EqualFold(*format, "vtt") && !strings.EqualFold(*format, "srt") {
		fail(errors.New("format must be vtt or srt"))
	}

	tmpWav, samplerate, err := toMono16k(*inPath)
	if err != nil {
		fail(err)
	}
	defer os.Remove(tmpWav)

	pcm, err := readWavPCM16(tmpWav, samplerate)
	if err != nil {
		fail(err)
	}
	if samplerate != 16000 {
		fail(fmt.Errorf("expected 16k WAV after decode, got %d", samplerate))
	}

	ctx, err := whisper.NewContext(*modelPath)
	if err != nil {
		fail(fmt.Errorf("load model: %w", err))
	}
	defer ctx.Close()

	if *threads > 0 {
		ctx.SetThreads(*threads)
	}

	segments := []whisper.Segment{}

	if *windowSec <= 0 {
		segs, err := transcribe(ctx, pcm, *lang)
		if err != nil {
			fail(err)
		}
		segments = append(segments, segs...)
	} else {
		win := *windowSec
		ovl := *overlapSec
		if win <= ovl {
			fail(errors.New("window must be > overlap"))
		}
		chunks := chunkPCM(pcm, 16000, win, ovl)
		offset := float32(0)
		for i, c := range chunks {
			segs, err := transcribe(ctx, c, *lang)
			if err != nil {
				fail(fmt.Errorf("chunk %d: %w", i, err))
			}
			// shift timestamps by offset
			for _, s := range segs {
				s.Start += offset
				s.End += offset
				segments = append(segments, s)
			}
			// advance offset by window - overlap
			offset += float32(win - ovl)
		}
		// de-duplicate potential overlap text:
		segments = dedupeOverlap(segments, float32(*overlapSec))
	}

	// Convert segments -> cues with neat wrapping
	cues := segmentsToCues(segments, *maxChars, *maxLines)

	// Write output
	var out *os.File
	if strings.ToLower(*outPath) == "-" {
		out = os.Stdout
	} else {
		_ = os.MkdirAll(filepath.Dir(*outPath), 0755)
		f, err := os.Create(*outPath)
		if err != nil {
			fail(err)
		}
		defer f.Close()
		out = f
	}

	switch strings.ToLower(*format) {
	case "vtt":
		if !strings.HasSuffix(strings.ToLower(*outPath), ".vtt") && *outPath != "-" {
			fmt.Fprintln(os.Stderr, "note: writing VTT; consider using .vtt extension")
		}
		writeVTT(out, cues)
	default:
		writeSRT(out, cues)
	}

	fmt.Fprintf(os.Stderr, "Wrote %d cues to %s\n", len(cues), *outPath)
}

// ---------- decoding (ffmpeg -> mono 16k WAV) ----------

func toMono16k(in string) (string, int, error) {
	tmp := filepath.Join(os.TempDir(), fmt.Sprintf("cap-%d.wav", time.Now().UnixNano()))
	cmd := exec.Command("ffmpeg",
		"-y", "-i", in,
		"-ac", "1",
		"-ar", "16000",
		"-acodec", "pcm_s16le",
		tmp,
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", 0, fmt.Errorf("ffmpeg: %v\n%s", err, stderr.String())
	}
	return tmp, 16000, nil
}

// read WAV PCM16 -> float32 mono samples
func readWavPCM16(path string, sampleRate int) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	// super-light WAV reader: skip 44-byte header
	if _, err := f.Seek(44, io.SeekStart); err != nil {
		return nil, err
	}
	r := bufio.NewReader(f)
	var samples []float32
	for {
		b0, err := r.ReadByte()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
		b1, err := r.ReadByte()
		if err != nil {
			return nil, err
		}
		// little-endian int16
		v := int16(uint16(b0) | uint16(b1)<<8)
		samples = append(samples, float32(v)/32768.0)
	}
	return samples, nil
}

// ---------- chunking ----------

func chunkPCM(pcm []float32, sr, windowSec, overlapSec int) [][]float32 {
	win := windowSec * sr
	ovl := overlapSec * sr
	if win <= 0 {
		return [][]float32{pcm}
	}
	var out [][]float32
	for start := 0; start < len(pcm); start += (win - ovl) {
		end := start + win
		if end > len(pcm) {
			end = len(pcm)
		}
		out = append(out, pcm[start:end])
		if end == len(pcm) {
			break
		}
	}
	return out
}

func dedupeOverlap(segs []whisper.Segment, overlap float32) []whisper.Segment {
	if len(segs) < 2 {
		return segs
	}
	out := []whisper.Segment{segs[0]}
	for i := 1; i < len(segs); i++ {
		prev := out[len(out)-1]
		cur := segs[i]
		// if same (or near-same) text within small time gap, drop duplicate
		if strings.TrimSpace(prev.Text) == strings.TrimSpace(cur.Text) &&
			math.Abs(float64(prev.Start-cur.Start)) < float64(overlap)+0.2 {
			continue
		}
		out = append(out, cur)
	}
	return out
}

// ---------- transcription ----------

func transcribe(ctx *whisper.Context, pcm []float32, language string) ([]whisper.Segment, error) {
	params := whisper.NewFullParams(whisper.SAMPLING_GREEDY)
	if language != "" {
		params.SetLanguage(language)
	} else {
		params.SetTranslate(false)
	}
	// Better timestamps / shorter chunks help avoid hallucinations:
	params.SetNoContext(true)
	params.SetSingleSegment(false)
	params.SetSuppressNonSpeechTokens(true)
	if err := ctx.Process(pcm, &params, nil, nil); err != nil {
		return nil, err
	}
	var segs []whisper.Segment
	ctx.ForEachSegment(func(s whisper.Segment) {
		// Trim whitespace early
		txt := strings.TrimSpace(s.Text)
		if txt != "" {
			s.Text = txt
			segs = append(segs, s)
		}
	})
	return segs, nil
}

// ---------- cue building & formatting ----------

type Cue struct {
	Idx       int
	Start     float32
	End       float32
	Lines     []string
	RawText   string
}

func segmentsToCues(segs []whisper.Segment, maxChars, maxLines int) []Cue {
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
	cues = mergeShortCues(cues, 0.6) // merge cues < 600 ms into neighbors when safe
	return cues
}

func wrapWords(s string, maxChars, maxLines int) []string {
	words := fieldsPreservePunct(s)
	var lines []string
	curr := ""
	for _, w := range words {
		if curr == "" {
			curr = w
			continue
		}
		if len(curr)+1+len(w) <= maxChars {
			curr += " " + w
		} else {
			lines = append(lines, curr)
			curr = w
			if len(lines) == maxLines-1 {
				// append remainder (even if it exceeds by a bit)
				rest := strings.Join(wordsJoinRemaining(words, w), " ")
				if rest != "" {
					if len(curr) > 0 {
						rest = curr + " " + rest
					} else {
						rest = curr
					}
					// ensure final line length cap-ish without splitting words
					lines = append(lines, softTruncate(rest, maxChars))
				}
				if len(lines) > maxLines {
					lines = lines[:maxLines]
				}
				return lines
			}
		}
	}
	if curr != "" {
		lines = append(lines, curr)
	}
	if len(lines) > maxLines {
		return lines[:maxLines]
	}
	return lines
}

func fieldsPreservePunct(s string) []string {
	// Split on whitespace only; keep punctuation attached to words to avoid mid-word breaks
	f := strings.Fields(s)
	return f
}

func wordsJoinRemaining(all []string, from string) []string {
	var rest []string
	found := false
	for _, w := range all {
		if !found && w == from {
			found = true
			continue
		}
		if found {
			rest = append(rest, w)
		}
	}
	return rest
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

func mergeShortCues(cues []Cue, minDur float64) []Cue {
	if len(cues) < 2 {
		return cues
	}
	var out []Cue
	for i := 0; i < len(cues); i++ {
		c := cues[i]
		dur := float64(c.End - c.Start)
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

func tsVTT(t float32) string {
	h := int(t) / 3600
	m := (int(t) % 3600) / 60
	s := t - float32(h*3600+m*60)
	return fmt.Sprintf("%02d:%02d:%06.3f", h, m, s)
}
func tsSRT(t float32) string {
	return strings.ReplaceAll(tsVTT(t), ".", ",")
}

func writeVTT(w io.Writer, cues []Cue) {
	fmt.Fprintln(w, "WEBVTT\n")
	for _, c := range cues {
		fmt.Fprintf(w, "%s --> %s\n", tsVTT(c.Start), tsVTT(c.End))
		for _, line := range c.Lines {
			fmt.Fprintln(w, line)
		}
		fmt.Fprintln(w)
	}
}

func writeSRT(w io.Writer, cues []Cue) {
	for _, c := range cues {
		fmt.Fprintln(w, c.Idx)
		fmt.Fprintf(w, "%s --> %s\n", tsSRT(c.Start), tsSRT(c.End))
		for _, line := range c.Lines {
			fmt.Fprintln(w, line)
		}
		fmt.Fprintln(w)
	}
}
