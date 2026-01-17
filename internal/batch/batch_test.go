package batch

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSupportedExtensions(t *testing.T) {
	// Check that common media formats are supported
	supported := []string{".mp3", ".mp4", ".wav", ".mkv", ".avi", ".mov", ".flac", ".ogg"}
	for _, ext := range supported {
		if !SupportedExtensions[ext] {
			t.Errorf("Extension %q should be supported", ext)
		}
	}

	// Check that non-media formats are not supported
	unsupported := []string{".txt", ".doc", ".pdf", ".jpg", ".png", ".exe"}
	for _, ext := range unsupported {
		if SupportedExtensions[ext] {
			t.Errorf("Extension %q should not be supported", ext)
		}
	}
}

func TestIsDirectory(t *testing.T) {
	// Create a temporary directory
	tmpDir := t.TempDir()

	// Test directory
	if !IsDirectory(tmpDir) {
		t.Errorf("IsDirectory(%q) = false, want true", tmpDir)
	}

	// Test file
	tmpFile := filepath.Join(tmpDir, "test.txt")
	if err := os.WriteFile(tmpFile, []byte("test"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}
	if IsDirectory(tmpFile) {
		t.Errorf("IsDirectory(%q) = true, want false", tmpFile)
	}

	// Test non-existent path
	if IsDirectory("/nonexistent/path") {
		t.Error("IsDirectory should return false for non-existent path")
	}
}

func TestScanDirectory(t *testing.T) {
	tmpDir := t.TempDir()

	// Create test files
	files := []string{
		"video1.mp4",
		"video2.mkv",
		"audio.mp3",
		"readme.txt",
		"image.png",
	}

	for _, f := range files {
		path := filepath.Join(tmpDir, f)
		if err := os.WriteFile(path, []byte("test"), 0644); err != nil {
			t.Fatalf("Failed to create test file: %v", err)
		}
	}

	// Scan directory
	jobs, err := ScanDirectory(tmpDir, "", "vtt", false)
	if err != nil {
		t.Fatalf("ScanDirectory failed: %v", err)
	}

	// Should find 3 media files (mp4, mkv, mp3)
	if len(jobs) != 3 {
		t.Errorf("ScanDirectory found %d files, want 3", len(jobs))
	}

	// Check job structure
	for _, job := range jobs {
		if job.Input == "" {
			t.Error("Job.Input should not be empty")
		}
		if job.Output == "" {
			t.Error("Job.Output should not be empty")
		}
		if job.BaseName == "" {
			t.Error("Job.BaseName should not be empty")
		}
		if !filepath.IsAbs(job.Input) {
			t.Errorf("Job.Input should be absolute path: %q", job.Input)
		}
	}
}

func TestScanDirectoryRecursive(t *testing.T) {
	tmpDir := t.TempDir()

	// Create subdirectory
	subDir := filepath.Join(tmpDir, "subdir")
	if err := os.MkdirAll(subDir, 0755); err != nil {
		t.Fatalf("Failed to create subdirectory: %v", err)
	}

	// Create test files
	if err := os.WriteFile(filepath.Join(tmpDir, "root.mp4"), []byte("test"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(subDir, "nested.mp4"), []byte("test"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Non-recursive scan
	jobs, err := ScanDirectory(tmpDir, "", "vtt", false)
	if err != nil {
		t.Fatalf("ScanDirectory failed: %v", err)
	}
	if len(jobs) != 1 {
		t.Errorf("Non-recursive scan found %d files, want 1", len(jobs))
	}

	// Recursive scan
	jobs, err = ScanDirectory(tmpDir, "", "vtt", true)
	if err != nil {
		t.Fatalf("ScanDirectory recursive failed: %v", err)
	}
	if len(jobs) != 2 {
		t.Errorf("Recursive scan found %d files, want 2", len(jobs))
	}
}

func TestScanDirectoryCustomOutput(t *testing.T) {
	tmpDir := t.TempDir()
	outDir := filepath.Join(tmpDir, "output")

	// Create test file
	if err := os.WriteFile(filepath.Join(tmpDir, "video.mp4"), []byte("test"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Scan with custom output directory
	jobs, err := ScanDirectory(tmpDir, outDir, "srt", false)
	if err != nil {
		t.Fatalf("ScanDirectory failed: %v", err)
	}

	if len(jobs) != 1 {
		t.Fatalf("Expected 1 job, got %d", len(jobs))
	}

	// Check output path uses custom directory and format
	if !filepath.HasPrefix(jobs[0].Output, outDir) {
		t.Errorf("Output path %q should be in %q", jobs[0].Output, outDir)
	}
	if filepath.Ext(jobs[0].Output) != ".srt" {
		t.Errorf("Output extension = %q, want .srt", filepath.Ext(jobs[0].Output))
	}
}

func TestScanDirectoryEmpty(t *testing.T) {
	tmpDir := t.TempDir()

	// Scan empty directory
	jobs, err := ScanDirectory(tmpDir, "", "vtt", false)
	if err != nil {
		t.Fatalf("ScanDirectory failed: %v", err)
	}

	if len(jobs) != 0 {
		t.Errorf("Empty directory should return 0 jobs, got %d", len(jobs))
	}
}
