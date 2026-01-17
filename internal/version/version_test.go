package version

import (
	"runtime"
	"strings"
	"testing"
)

func TestGet(t *testing.T) {
	info := Get()

	// GoVersion should always be populated
	if info.GoVersion == "" {
		t.Error("GoVersion should not be empty")
	}

	// Should start with "go"
	if !strings.HasPrefix(info.GoVersion, "go") {
		t.Errorf("GoVersion = %q, want prefix 'go'", info.GoVersion)
	}

	// OS and Arch should match runtime
	if info.OS != runtime.GOOS {
		t.Errorf("OS = %q, want %q", info.OS, runtime.GOOS)
	}
	if info.Arch != runtime.GOARCH {
		t.Errorf("Arch = %q, want %q", info.Arch, runtime.GOARCH)
	}
}

func TestInfoString(t *testing.T) {
	info := Info{
		Version:   "1.0.0",
		Commit:    "abc1234",
		Date:      "2024-01-01",
		GoVersion: "go1.23",
		OS:        "linux",
		Arch:      "amd64",
	}

	str := info.String()

	// Should contain version
	if !strings.Contains(str, "1.0.0") {
		t.Errorf("String() should contain version, got: %s", str)
	}

	// Should contain commit
	if !strings.Contains(str, "abc1234") {
		t.Errorf("String() should contain commit, got: %s", str)
	}

	// Should contain Go version
	if !strings.Contains(str, "go1.23") {
		t.Errorf("String() should contain GoVersion, got: %s", str)
	}

	// Should contain OS/Arch
	if !strings.Contains(str, "linux/amd64") {
		t.Errorf("String() should contain OS/Arch, got: %s", str)
	}
}

func TestInfoShort(t *testing.T) {
	info := Info{
		Version: "1.0.0",
		Commit:  "abc1234",
	}

	short := info.Short()

	if short != "1.0.0 (abc1234)" {
		t.Errorf("Short() = %q, want %q", short, "1.0.0 (abc1234)")
	}
}

func TestDefaultValues(t *testing.T) {
	// When not set at build time, should have default values
	if Version != "dev" {
		// This might be set by build, which is fine
		t.Logf("Version = %q (may be set by build)", Version)
	}
}
