package batch

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// SupportedExtensions lists audio/video file extensions that can be processed.
var SupportedExtensions = map[string]bool{
	".mp3":  true,
	".mp4":  true,
	".m4a":  true,
	".wav":  true,
	".flac": true,
	".ogg":  true,
	".webm": true,
	".mkv":  true,
	".avi":  true,
	".mov":  true,
	".aac":  true,
	".wma":  true,
	".opus": true,
}

// FileJob represents a single file to process in batch mode.
type FileJob struct {
	Input    string
	Output   string
	BaseName string
}

// ScanDirectory finds all supported audio/video files in a directory.
func ScanDirectory(dir string, outputDir string, outputFormat string, recursive bool) ([]FileJob, error) {
	var jobs []FileJob
	outputs := make(map[string]string)
	outputFormat = strings.ToLower(outputFormat)

	if outputDir == "" {
		outputDir = dir
	}

	walkFn := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories unless we're being recursive
		if info.IsDir() {
			if path != dir && !recursive {
				return filepath.SkipDir
			}
			return nil
		}

		// Check if file has a supported extension
		ext := filepath.Ext(path)
		if !SupportedExtensions[strings.ToLower(ext)] {
			return nil
		}

		// Create output path
		baseName := strings.TrimSuffix(filepath.Base(path), ext)
		relPath, err := filepath.Rel(dir, filepath.Dir(path))
		if err != nil {
			return err
		}
		outDir := filepath.Join(outputDir, relPath)
		outPath := filepath.Join(outDir, baseName+"."+outputFormat)
		if previous, exists := outputs[outPath]; exists {
			return fmt.Errorf("inputs %q and %q map to the same caption file %q; rename one input", previous, path, outPath)
		}
		outputs[outPath] = path

		jobs = append(jobs, FileJob{
			Input:    path,
			Output:   outPath,
			BaseName: baseName,
		})

		return nil
	}

	if err := filepath.Walk(dir, walkFn); err != nil {
		return nil, err
	}

	return jobs, nil
}

// IsDirectory checks if a path is a directory.
func IsDirectory(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return info.IsDir()
}
