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

	"opencaption/internal/batch"
	"opencaption/internal/captions"
	"opencaption/internal/chunk"
	"opencaption/internal/config"
	"opencaption/internal/decode"
	"opencaption/internal/device"
	"opencaption/internal/format"
	"opencaption/internal/models"
	"opencaption/internal/transcribe"
	"opencaption/internal/vad"
)

// ---------- CLI flags ----------
var (
	inPath     = flag.String("in", "", "Input audio/video file or directory (for batch mode)")
	outPath    = flag.String("out", "captions.vtt", "Output captions file (.vtt, .srt, .json) or directory")
	modelPath  = flag.String("model", "whisper.cpp/models/ggml-base.en.bin", "Path to ggml/gguf model or model name")
	modelDir   = flag.String("model-dir", "", "Directory for model files (enables auto-download)")
	ffmpegPath = flag.String("ffmpeg-path", "ffmpeg", "Path to ffmpeg binary")
	deviceFlag = flag.String("device", "auto", "Compute device: auto | cpu | gpu")
	useVAD     = flag.Bool("vad", true, "Enable simple energy-based VAD")
	lang       = flag.String("lang", "", "Language (e.g. 'en'); empty = auto")
	windowSec  = flag.Int("window", 60, "Chunk window seconds (0 = whole file)")
	overlapSec = flag.Int("overlap", 1, "Chunk overlap seconds (only if window > 0)")
	maxChars   = flag.Int("maxchars", 42, "Max characters per line")
	maxLines   = flag.Int("maxlines", 2, "Max lines per cue")
	formatFlag = flag.String("format", "vtt", "Caption format: vtt | srt | json")
	threads    = flag.Int("threads", 0, "Threads (0 = auto)")
	configFile = flag.String("config", "", "Path to config file (.json or .yaml)")
	preset     = flag.String("preset", "", "Use preset: fast | accurate | subtitle")
	batchMode  = flag.Bool("batch", false, "Process all media files in input directory")
	recursive  = flag.Bool("recursive", false, "Process subdirectories in batch mode")
)

func fail(err error) {
	fmt.Fprintln(os.Stderr, "Error:", err)
	os.Exit(1)
}

func main() {
	flag.Parse()

	// Load config file if specified
	var cfg *config.Config
	if *configFile != "" {
		var err error
		cfg, err = config.LoadFile(*configFile)
		if err != nil {
			fail(err)
		}
	} else {
		cfg = config.DefaultConfig()
	}

	// Apply preset if specified
	if *preset != "" {
		if err := cfg.ApplyPreset(*preset); err != nil {
			fail(err)
		}
	}

	// Override config with command-line flags (flags take precedence)
	applyFlags(cfg)

	// Validate input
	if cfg.Input == "" {
		fail(errors.New("please provide -in <file> or -in <directory> with -batch"))
	}

	// Validate format
	formatLower := strings.ToLower(cfg.Format)
	if formatLower != "vtt" && formatLower != "srt" && formatLower != "json" {
		fail(errors.New("format must be vtt, srt, or json"))
	}

	// Handle device detection
	devType, err := device.ParseType(cfg.Device)
	if err != nil {
		fail(err)
	}
	devInfo := device.Detect(devType)
	device.SetEnvironment(devInfo.Type)
	if devInfo.Fallback {
		fmt.Fprintf(os.Stderr, "Note: GPU not available, using %s\n", devInfo.Name)
	} else {
		fmt.Fprintf(os.Stderr, "Using device: %s\n", devInfo.Name)
	}

	// Resolve model path (with auto-download if model-dir is set)
	resolvedModel, err := models.ResolveModel(cfg.Model, cfg.ModelDir, cfg.ModelDir != "")
	if err != nil {
		fail(err)
	}
	cfg.Model = resolvedModel

	// Check for batch mode
	if cfg.BatchMode || batch.IsDirectory(cfg.Input) {
		if !batch.IsDirectory(cfg.Input) {
			fail(errors.New("batch mode requires -in to be a directory"))
		}
		runBatch(cfg)
		return
	}

	// Single file processing
	cues, err := processFile(cfg)
	if err != nil {
		fail(err)
	}

	// Write output
	if err := writeOutput(cfg.Output, cfg.Format, cues); err != nil {
		fail(err)
	}

	fmt.Fprintf(os.Stderr, "Wrote %d cues to %s\n", len(cues), cfg.Output)
}

func applyFlags(cfg *config.Config) {
	// Apply command-line flags to config (only if flag was explicitly set)
	flag.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "in":
			cfg.Input = *inPath
		case "out":
			cfg.Output = *outPath
		case "model":
			cfg.Model = *modelPath
		case "model-dir":
			cfg.ModelDir = *modelDir
		case "ffmpeg-path":
			cfg.FFmpegPath = *ffmpegPath
		case "device":
			cfg.Device = *deviceFlag
		case "vad":
			cfg.VAD = *useVAD
		case "lang":
			cfg.Language = *lang
		case "window":
			cfg.Window = *windowSec
		case "overlap":
			cfg.Overlap = *overlapSec
		case "maxchars":
			cfg.MaxChars = *maxChars
		case "maxlines":
			cfg.MaxLines = *maxLines
		case "format":
			cfg.Format = *formatFlag
		case "threads":
			cfg.Threads = *threads
		case "batch":
			cfg.BatchMode = *batchMode
		case "recursive":
			cfg.Recursive = *recursive
		}
	})

	// Also apply defaults from flags if config values are zero
	if cfg.Input == "" {
		cfg.Input = *inPath
	}
	if cfg.Output == "" {
		cfg.Output = *outPath
	}
}

func processFile(cfg *config.Config) ([]captions.Cue, error) {
	pcm, err := decode.DecodeToPCM16(cfg.Input, cfg.FFmpegPath)
	if err != nil {
		return nil, err
	}

	model, err := whisper.New(cfg.Model)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	defer model.Close()

	ctx, err := model.NewContext()
	if err != nil {
		return nil, fmt.Errorf("create context: %w", err)
	}

	if cfg.Threads > 0 {
		ctx.SetThreads(uint(cfg.Threads))
	}

	const sampleRate = 16000
	var segments []whisper.Segment

	var speechSegs []vad.Segment
	if cfg.VAD {
		speechSegs = vad.SpeechSegments(pcm, sampleRate)
	} else {
		speechSegs = []vad.Segment{{Start: 0, End: len(pcm)}}
	}
	if len(speechSegs) == 0 {
		speechSegs = []vad.Segment{{Start: 0, End: len(pcm)}}
	}

	if cfg.Window > 0 && cfg.Window <= cfg.Overlap {
		return nil, errors.New("window must be > overlap")
	}

	for _, seg := range speechSegs {
		if seg.End <= seg.Start || seg.Start < 0 || seg.End > len(pcm) {
			continue
		}
		segPCM := pcm[seg.Start:seg.End]
		baseOffset := time.Duration(float64(seg.Start) / float64(sampleRate) * float64(time.Second))

		if cfg.Window <= 0 {
			segs, err := transcribe.Transcribe(ctx, segPCM, cfg.Language)
			if err != nil {
				return nil, err
			}
			for _, s := range segs {
				s.Start += baseOffset
				s.End += baseOffset
				segments = append(segments, s)
			}
			continue
		}

		chunks := chunk.ChunkPCM(segPCM, sampleRate, cfg.Window, cfg.Overlap)
		for i, c := range chunks {
			segs, err := transcribe.Transcribe(ctx, c.PCM, cfg.Language)
			if err != nil {
				return nil, fmt.Errorf("chunk %d: %w", i, err)
			}
			chunkOffset := baseOffset + time.Duration(float64(c.Start)/float64(sampleRate)*float64(time.Second))
			for _, s := range segs {
				s.Start += chunkOffset
				s.End += chunkOffset
				segments = append(segments, s)
			}
		}
	}

	// De-duplicate potential overlap text
	segments = chunk.DedupeOverlap(segments, time.Duration(cfg.Overlap)*time.Second)

	// Convert segments -> cues with neat wrapping
	return captions.SegmentsToCues(segments, cfg.MaxChars, cfg.MaxLines), nil
}

func writeOutput(outPath, formatType string, cues []captions.Cue) error {
	var out *os.File
	if strings.ToLower(outPath) == "-" {
		out = os.Stdout
	} else {
		if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil && !os.IsExist(err) {
			return err
		}
		f, err := os.Create(outPath)
		if err != nil {
			return err
		}
		defer f.Close()
		out = f
	}

	switch strings.ToLower(formatType) {
	case "json":
		return format.WriteJSON(out, cues)
	case "srt":
		format.WriteSRT(out, cues)
	default:
		if !strings.HasSuffix(strings.ToLower(outPath), ".vtt") && outPath != "-" {
			fmt.Fprintln(os.Stderr, "note: writing VTT; consider using .vtt extension")
		}
		format.WriteVTT(out, cues)
	}
	return nil
}

func runBatch(cfg *config.Config) {
	// Determine output directory
	outDir := cfg.Output
	if !batch.IsDirectory(outDir) {
		// If output is a file pattern, use input directory
		outDir = cfg.Input
	}

	jobs, err := batch.ScanDirectory(cfg.Input, outDir, cfg.Format, cfg.Recursive)
	if err != nil {
		fail(err)
	}

	if len(jobs) == 0 {
		fmt.Fprintln(os.Stderr, "No supported media files found in", cfg.Input)
		return
	}

	fmt.Fprintf(os.Stderr, "Found %d files to process\n", len(jobs))

	// Load model once for all files
	model, err := whisper.New(cfg.Model)
	if err != nil {
		fail(fmt.Errorf("load model: %w", err))
	}
	defer model.Close()

	successCount := 0
	for i, job := range jobs {
		fmt.Fprintf(os.Stderr, "[%d/%d] Processing %s...\n", i+1, len(jobs), job.BaseName)

		// Create a copy of config for this job
		jobCfg := *cfg
		jobCfg.Input = job.Input
		jobCfg.Output = job.Output

		cues, err := processFileWithModel(&jobCfg, model)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Error: %v\n", err)
			continue
		}

		if err := writeOutput(job.Output, cfg.Format, cues); err != nil {
			fmt.Fprintf(os.Stderr, "  Error writing output: %v\n", err)
			continue
		}

		fmt.Fprintf(os.Stderr, "  Wrote %d cues to %s\n", len(cues), job.Output)
		successCount++
	}

	fmt.Fprintf(os.Stderr, "\nBatch complete: %d/%d files processed successfully\n", successCount, len(jobs))
}

func processFileWithModel(cfg *config.Config, model whisper.Model) ([]captions.Cue, error) {
	pcm, err := decode.DecodeToPCM16(cfg.Input, cfg.FFmpegPath)
	if err != nil {
		return nil, err
	}

	ctx, err := model.NewContext()
	if err != nil {
		return nil, fmt.Errorf("create context: %w", err)
	}

	if cfg.Threads > 0 {
		ctx.SetThreads(uint(cfg.Threads))
	}

	const sampleRate = 16000
	var segments []whisper.Segment

	var speechSegs []vad.Segment
	if cfg.VAD {
		speechSegs = vad.SpeechSegments(pcm, sampleRate)
	} else {
		speechSegs = []vad.Segment{{Start: 0, End: len(pcm)}}
	}
	if len(speechSegs) == 0 {
		speechSegs = []vad.Segment{{Start: 0, End: len(pcm)}}
	}

	if cfg.Window > 0 && cfg.Window <= cfg.Overlap {
		return nil, errors.New("window must be > overlap")
	}

	for _, seg := range speechSegs {
		if seg.End <= seg.Start || seg.Start < 0 || seg.End > len(pcm) {
			continue
		}
		segPCM := pcm[seg.Start:seg.End]
		baseOffset := time.Duration(float64(seg.Start) / float64(sampleRate) * float64(time.Second))

		if cfg.Window <= 0 {
			segs, err := transcribe.Transcribe(ctx, segPCM, cfg.Language)
			if err != nil {
				return nil, err
			}
			for _, s := range segs {
				s.Start += baseOffset
				s.End += baseOffset
				segments = append(segments, s)
			}
			continue
		}

		chunks := chunk.ChunkPCM(segPCM, sampleRate, cfg.Window, cfg.Overlap)
		for i, c := range chunks {
			segs, err := transcribe.Transcribe(ctx, c.PCM, cfg.Language)
			if err != nil {
				return nil, fmt.Errorf("chunk %d: %w", i, err)
			}
			chunkOffset := baseOffset + time.Duration(float64(c.Start)/float64(sampleRate)*float64(time.Second))
			for _, s := range segs {
				s.Start += chunkOffset
				s.End += chunkOffset
				segments = append(segments, s)
			}
		}
	}

	// De-duplicate potential overlap text
	segments = chunk.DedupeOverlap(segments, time.Duration(cfg.Overlap)*time.Second)

	// Convert segments -> cues with neat wrapping
	return captions.SegmentsToCues(segments, cfg.MaxChars, cfg.MaxLines), nil
}
