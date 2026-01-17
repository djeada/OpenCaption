package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"syscall"
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
	"opencaption/internal/version"
)

// ---------- CLI flags ----------
var (
	inPath      = flag.String("in", "", "Input audio/video file or directory (for batch mode)")
	outPath     = flag.String("out", "captions.vtt", "Output captions file (.vtt, .srt, .json) or directory")
	modelPath   = flag.String("model", "whisper.cpp/models/ggml-base.en.bin", "Path to ggml/gguf model or model name")
	modelDir    = flag.String("model-dir", "", "Directory for model files (enables auto-download)")
	ffmpegPath  = flag.String("ffmpeg-path", "ffmpeg", "Path to ffmpeg binary")
	deviceFlag  = flag.String("device", "auto", "Compute device: auto | cpu | gpu")
	useVAD      = flag.Bool("vad", true, "Enable simple energy-based VAD")
	lang        = flag.String("lang", "", "Language (e.g. 'en'); empty = auto")
	windowSec   = flag.Int("window", 60, "Chunk window seconds (0 = whole file)")
	overlapSec  = flag.Int("overlap", 1, "Chunk overlap seconds (only if window > 0)")
	maxChars    = flag.Int("maxchars", 42, "Max characters per line")
	maxLines    = flag.Int("maxlines", 2, "Max lines per cue")
	formatFlag  = flag.String("format", "vtt", "Caption format: vtt | srt | json")
	threads     = flag.Int("threads", 0, "Threads (0 = auto)")
	configFile  = flag.String("config", "", "Path to config file (.json or .yaml)")
	preset      = flag.String("preset", "", "Use preset: fast | accurate | subtitle")
	batchMode   = flag.Bool("batch", false, "Process all media files in input directory")
	recursive   = flag.Bool("recursive", false, "Process subdirectories in batch mode")
	showVersion = flag.Bool("version", false, "Show version information")
	verbose     = flag.Bool("verbose", false, "Enable verbose output")
	quiet       = flag.Bool("quiet", false, "Suppress non-error output")
	workers     = flag.Int("workers", 0, "Number of parallel workers for batch mode (0 = auto)")
	listModels  = flag.Bool("list-models", false, "List available models")
)

var logger *slog.Logger

func fail(err error) {
	if logger != nil {
		logger.Error("fatal error", slog.String("error", err.Error()))
	} else {
		fmt.Fprintln(os.Stderr, "Error:", err)
	}
	os.Exit(1)
}

func main() {
	flag.Parse()

	// Handle special flags first
	if *showVersion {
		fmt.Println(version.Get().String())
		return
	}

	if *listModels {
		printModelList()
		return
	}

	// Setup logging
	logLevel := slog.LevelInfo
	if *verbose {
		logLevel = slog.LevelDebug
	}
	if *quiet {
		logLevel = slog.LevelError
	}
	logger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: logLevel}))
	slog.SetDefault(logger)

	// Setup signal handling for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		logger.Info("received interrupt signal, shutting down...")
		cancel()
	}()

	// Load config file if specified
	var cfg *config.Config
	if *configFile != "" {
		var err error
		cfg, err = config.LoadFile(*configFile)
		if err != nil {
			fail(err)
		}
		logger.Debug("loaded config file", slog.String("path", *configFile))
	} else {
		cfg = config.DefaultConfig()
	}

	// Apply preset if specified
	if *preset != "" {
		if err := cfg.ApplyPreset(*preset); err != nil {
			fail(err)
		}
		logger.Debug("applied preset", slog.String("preset", *preset))
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
		logger.Warn("GPU not available, using fallback", slog.String("device", devInfo.Name))
	} else {
		logger.Info("using device", slog.String("device", devInfo.Name))
	}

	// Resolve model path (with auto-download if model-dir is set)
	resolvedModel, err := models.ResolveModelWithContext(ctx, cfg.Model, cfg.ModelDir, cfg.ModelDir != "")
	if err != nil {
		fail(err)
	}
	cfg.Model = resolvedModel
	logger.Debug("using model", slog.String("path", resolvedModel))

	// Check for batch mode
	if cfg.BatchMode || batch.IsDirectory(cfg.Input) {
		if !batch.IsDirectory(cfg.Input) {
			fail(errors.New("batch mode requires -in to be a directory"))
		}
		runBatch(ctx, cfg)
		return
	}

	// Single file processing
	cues, err := processFile(ctx, cfg)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			logger.Info("processing cancelled")
			return
		}
		fail(err)
	}

	// Write output
	if err := writeOutput(cfg.Output, cfg.Format, cues); err != nil {
		fail(err)
	}

	logger.Info("completed", slog.Int("cues", len(cues)), slog.String("output", cfg.Output))
}

func printModelList() {
	fmt.Println("Available Whisper models:")
	fmt.Println()
	fmt.Printf("%-18s %-10s %-12s %s\n", "NAME", "SIZE", "MULTILINGUAL", "DESCRIPTION")
	fmt.Println(strings.Repeat("-", 70))

	// Get sorted list of models from catalog
	var modelNames []string
	for name := range models.ModelCatalog {
		modelNames = append(modelNames, name)
	}
	sort.Strings(modelNames)

	for _, name := range modelNames {
		info := models.ModelCatalog[name]
		multi := "Yes"
		if !info.Multilingual {
			multi = "No"
		}
		fmt.Printf("%-18s %-10s %-12s %s\n", info.Name, info.Size, multi, info.Description)
	}

	fmt.Println()
	fmt.Println("Use --model <name> --model-dir <path> to auto-download a model.")
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

func processFile(ctx context.Context, cfg *config.Config) ([]captions.Cue, error) {
	// Check for cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	pcm, err := decode.DecodeToPCM16(cfg.Input, cfg.FFmpegPath)
	if err != nil {
		return nil, err
	}

	model, err := whisper.New(cfg.Model)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	defer model.Close()

	wctx, err := model.NewContext()
	if err != nil {
		return nil, fmt.Errorf("create context: %w", err)
	}

	if cfg.Threads > 0 {
		wctx.SetThreads(uint(cfg.Threads))
	}

	return processAudio(ctx, wctx, pcm, cfg)
}

func processAudio(ctx context.Context, wctx whisper.Context, pcm []float32, cfg *config.Config) ([]captions.Cue, error) {
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
		// Check for cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		if seg.End <= seg.Start || seg.Start < 0 || seg.End > len(pcm) {
			continue
		}
		segPCM := pcm[seg.Start:seg.End]
		baseOffset := time.Duration(float64(seg.Start) / float64(sampleRate) * float64(time.Second))

		if cfg.Window <= 0 {
			segs, err := transcribe.Transcribe(wctx, segPCM, cfg.Language)
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
			segs, err := transcribe.Transcribe(wctx, c.PCM, cfg.Language)
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
			logger.Warn("writing VTT format; consider using .vtt extension")
		}
		format.WriteVTT(out, cues)
	}
	return nil
}

func runBatch(ctx context.Context, cfg *config.Config) {
	// Determine output directory
	outDir := cfg.Output
	if !batch.IsDirectory(outDir) {
		outDir = cfg.Input
	}

	jobs, err := batch.ScanDirectory(cfg.Input, outDir, cfg.Format, cfg.Recursive)
	if err != nil {
		fail(err)
	}

	if len(jobs) == 0 {
		logger.Warn("no supported media files found", slog.String("directory", cfg.Input))
		return
	}

	logger.Info("starting batch processing", slog.Int("files", len(jobs)))

	// Load model once for all files
	model, err := whisper.New(cfg.Model)
	if err != nil {
		fail(fmt.Errorf("load model: %w", err))
	}
	defer model.Close()

	// Determine number of workers
	// Default cap is set to avoid memory issues - large Whisper models can use 1-3GB+ RAM each
	const defaultMaxWorkers = 4
	numWorkers := *workers
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
		if numWorkers > defaultMaxWorkers {
			numWorkers = defaultMaxWorkers
		}
	}

	// For single worker, process sequentially
	if numWorkers == 1 {
		runBatchSequential(ctx, cfg, model, jobs)
		return
	}

	// Parallel processing with worker pool
	runBatchParallel(ctx, cfg, model, jobs, numWorkers)
}

func runBatchSequential(ctx context.Context, cfg *config.Config, model whisper.Model, jobs []batch.FileJob) {
	successCount := 0
	for i, job := range jobs {
		select {
		case <-ctx.Done():
			logger.Info("batch processing cancelled")
			return
		default:
		}

		logger.Info("processing file",
			slog.Int("current", i+1),
			slog.Int("total", len(jobs)),
			slog.String("file", job.BaseName))

		jobCfg := *cfg
		jobCfg.Input = job.Input
		jobCfg.Output = job.Output

		cues, err := processFileWithModel(ctx, &jobCfg, model)
		if err != nil {
			logger.Error("processing failed",
				slog.String("file", job.BaseName),
				slog.String("error", err.Error()))
			continue
		}

		if err := writeOutput(job.Output, cfg.Format, cues); err != nil {
			logger.Error("write failed",
				slog.String("file", job.BaseName),
				slog.String("error", err.Error()))
			continue
		}

		logger.Debug("completed file",
			slog.String("file", job.BaseName),
			slog.Int("cues", len(cues)))
		successCount++
	}

	logger.Info("batch complete",
		slog.Int("success", successCount),
		slog.Int("total", len(jobs)))
}

func runBatchParallel(ctx context.Context, cfg *config.Config, model whisper.Model, jobs []batch.FileJob, numWorkers int) {
	logger.Info("using parallel processing", slog.Int("workers", numWorkers))

	type result struct {
		job   batch.FileJob
		cues  []captions.Cue
		err   error
		index int
	}

	jobChan := make(chan struct {
		job   batch.FileJob
		index int
	}, len(jobs))
	resultChan := make(chan result, len(jobs))

	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range jobChan {
				select {
				case <-ctx.Done():
					resultChan <- result{job: item.job, index: item.index, err: ctx.Err()}
					continue
				default:
				}

				jobCfg := *cfg
				jobCfg.Input = item.job.Input
				jobCfg.Output = item.job.Output

				cues, err := processFileWithModel(ctx, &jobCfg, model)
				resultChan <- result{job: item.job, cues: cues, err: err, index: item.index}
			}
		}()
	}

	// Send jobs
	go func() {
		for i, job := range jobs {
			jobChan <- struct {
				job   batch.FileJob
				index int
			}{job: job, index: i}
		}
		close(jobChan)
	}()

	// Wait for workers and close results
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	successCount := 0
	for res := range resultChan {
		if res.err != nil {
			if !errors.Is(res.err, context.Canceled) {
				logger.Error("processing failed",
					slog.String("file", res.job.BaseName),
					slog.String("error", res.err.Error()))
			}
			continue
		}

		if err := writeOutput(res.job.Output, cfg.Format, res.cues); err != nil {
			logger.Error("write failed",
				slog.String("file", res.job.BaseName),
				slog.String("error", err.Error()))
			continue
		}

		logger.Debug("completed file",
			slog.String("file", res.job.BaseName),
			slog.Int("cues", len(res.cues)))
		successCount++
	}

	logger.Info("batch complete",
		slog.Int("success", successCount),
		slog.Int("total", len(jobs)))
}

func processFileWithModel(ctx context.Context, cfg *config.Config, model whisper.Model) ([]captions.Cue, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	pcm, err := decode.DecodeToPCM16(cfg.Input, cfg.FFmpegPath)
	if err != nil {
		return nil, err
	}

	wctx, err := model.NewContext()
	if err != nil {
		return nil, fmt.Errorf("create context: %w", err)
	}

	if cfg.Threads > 0 {
		wctx.SetThreads(uint(cfg.Threads))
	}

	return processAudio(ctx, wctx, pcm, cfg)
}
