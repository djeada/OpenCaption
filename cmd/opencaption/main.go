package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"

	"opencaption/internal/captions"
	"opencaption/internal/chunk"
	"opencaption/internal/decode"
	"opencaption/internal/format"
	"opencaption/internal/transcribe"
	"opencaption/internal/vad"
)

// ---------- CLI flags ----------
var (
	inPath     = flag.String("in", "", "Input audio/video file")
	outPath    = flag.String("out", "captions.vtt", "Output captions file (.vtt or .srt)")
	modelPath  = flag.String("model", "whisper.cpp/models/ggml-base.en.bin", "Path to ggml/gguf model")
	ffmpegPath = flag.String("ffmpeg-path", "ffmpeg", "Path to ffmpeg binary")
	useVAD     = flag.Bool("vad", true, "Enable simple energy-based VAD")
	lang       = flag.String("lang", "", "Language (e.g. 'en'); empty = auto")
	windowSec  = flag.Int("window", 60, "Chunk window seconds (0 = whole file)")
	overlapSec = flag.Int("overlap", 1, "Chunk overlap seconds (only if window > 0)")
	maxChars   = flag.Int("maxchars", 42, "Max characters per line")
	maxLines   = flag.Int("maxlines", 2, "Max lines per cue")
	formatFlag = flag.String("format", "vtt", "Caption format: vtt | srt (default vtt)")
	threads    = flag.Int("threads", 0, "Threads (0 = auto)")
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
	if !strings.EqualFold(*formatFlag, "vtt") && !strings.EqualFold(*formatFlag, "srt") {
		fail(errors.New("format must be vtt or srt"))
	}

	pcm, err := decode.DecodeToPCM16(*inPath, *ffmpegPath)
	if err != nil {
		fail(err)
	}

	model, err := whisper.New(*modelPath)
	if err != nil {
		fail(fmt.Errorf("load model: %w", err))
	}
	defer model.Close()

	ctx, err := model.NewContext()
	if err != nil {
		fail(fmt.Errorf("create context: %w", err))
	}

	if *threads > 0 {
		ctx.SetThreads(uint(*threads))
	}

	const sampleRate = 16000
	segments := []whisper.Segment{}

	var speechSegs []vad.Segment
	if *useVAD {
		speechSegs = vad.SpeechSegments(pcm, sampleRate)
	} else {
		speechSegs = []vad.Segment{{Start: 0, End: len(pcm)}}
	}
	if len(speechSegs) == 0 {
		speechSegs = []vad.Segment{{Start: 0, End: len(pcm)}}
	}

	if *windowSec > 0 && *windowSec <= *overlapSec {
		fail(errors.New("window must be > overlap"))
	}

	for _, seg := range speechSegs {
		if seg.End <= seg.Start || seg.Start < 0 || seg.End > len(pcm) {
			continue
		}
		segPCM := pcm[seg.Start:seg.End]
		baseOffset := time.Duration(float64(seg.Start) / float64(sampleRate) * float64(time.Second))

		if *windowSec <= 0 {
			segs, err := transcribe.Transcribe(ctx, segPCM, *lang)
			if err != nil {
				fail(err)
			}
			for _, s := range segs {
				s.Start += baseOffset
				s.End += baseOffset
				segments = append(segments, s)
			}
			continue
		}

		win := *windowSec
		ovl := *overlapSec
		chunks := chunk.ChunkPCM(segPCM, sampleRate, win, ovl)
		for i, c := range chunks {
			segs, err := transcribe.Transcribe(ctx, c.PCM, *lang)
			if err != nil {
				fail(fmt.Errorf("chunk %d: %w", i, err))
			}
			chunkOffset := baseOffset + time.Duration(float64(c.Start)/float64(sampleRate)*float64(time.Second))
			for _, s := range segs {
				s.Start += chunkOffset
				s.End += chunkOffset
				segments = append(segments, s)
			}
		}
	}
	// de-duplicate potential overlap text
	segments = chunk.DedupeOverlap(segments, time.Duration(*overlapSec)*time.Second)

	// Convert segments -> cues with neat wrapping
	cues := captions.SegmentsToCues(segments, *maxChars, *maxLines)

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

	switch strings.ToLower(*formatFlag) {
	case "vtt":
		if !strings.HasSuffix(strings.ToLower(*outPath), ".vtt") && *outPath != "-" {
			fmt.Fprintln(os.Stderr, "note: writing VTT; consider using .vtt extension")
		}
		format.WriteVTT(out, cues)
	default:
		format.WriteSRT(out, cues)
	}

	fmt.Fprintf(os.Stderr, "Wrote %d cues to %s\n", len(cues), *outPath)
}
