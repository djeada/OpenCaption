// Package version provides build-time version information.
package version

import (
	"fmt"
	"runtime"
	"runtime/debug"
)

// These variables are set at build time using ldflags.
var (
	// Version is the semantic version of the application.
	Version = "dev"
	// Commit is the git commit hash.
	Commit = "unknown"
	// Date is the build date.
	Date = "unknown"
)

// Info contains version information.
type Info struct {
	Version   string `json:"version"`
	Commit    string `json:"commit"`
	Date      string `json:"date"`
	GoVersion string `json:"go_version"`
	OS        string `json:"os"`
	Arch      string `json:"arch"`
}

// Get returns the current version information.
func Get() Info {
	info := Info{
		Version:   Version,
		Commit:    Commit,
		Date:      Date,
		GoVersion: runtime.Version(),
		OS:        runtime.GOOS,
		Arch:      runtime.GOARCH,
	}

	// Try to get version from build info if not set
	if info.Version == "dev" {
		if bi, ok := debug.ReadBuildInfo(); ok {
			if bi.Main.Version != "" && bi.Main.Version != "(devel)" {
				info.Version = bi.Main.Version
			}
			for _, setting := range bi.Settings {
				switch setting.Key {
				case "vcs.revision":
					if info.Commit == "unknown" && len(setting.Value) >= 7 {
						info.Commit = setting.Value[:7]
					}
				case "vcs.time":
					if info.Date == "unknown" {
						info.Date = setting.Value
					}
				}
			}
		}
	}

	return info
}

// String returns a human-readable version string.
func (i Info) String() string {
	return fmt.Sprintf("opencaption %s (%s) built %s with %s for %s/%s",
		i.Version, i.Commit, i.Date, i.GoVersion, i.OS, i.Arch)
}

// Short returns a short version string.
func (i Info) Short() string {
	return fmt.Sprintf("%s (%s)", i.Version, i.Commit)
}
