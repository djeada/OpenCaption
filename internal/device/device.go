package device

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

// Type represents the compute device type.
type Type string

const (
	Auto Type = "auto"
	CPU  Type = "cpu"
	GPU  Type = "gpu"
)

// Info contains information about the selected device.
type Info struct {
	Type       Type
	Name       string
	Available  bool
	Fallback   bool
	FallbackTo Type
}

// Detect determines what compute devices are available.
func Detect(requested Type) Info {
	info := Info{Type: requested, Available: true}

	switch requested {
	case CPU:
		info.Name = fmt.Sprintf("CPU (%s, %d cores)", runtime.GOARCH, runtime.NumCPU())
		return info

	case GPU:
		if gpuInfo := detectGPU(); gpuInfo != "" {
			info.Name = gpuInfo
			return info
		}
		// GPU requested but not available - fallback
		info.Available = false
		info.Fallback = true
		info.FallbackTo = CPU
		info.Name = fmt.Sprintf("CPU (%s, %d cores) [GPU not available]", runtime.GOARCH, runtime.NumCPU())
		return info

	case Auto:
		fallthrough
	default:
		// Auto: try GPU first, fall back to CPU
		if gpuInfo := detectGPU(); gpuInfo != "" {
			info.Type = GPU
			info.Name = gpuInfo
			return info
		}
		info.Type = CPU
		info.Name = fmt.Sprintf("CPU (%s, %d cores)", runtime.GOARCH, runtime.NumCPU())
		return info
	}
}

// detectGPU checks for available GPU compute resources.
func detectGPU() string {
	// Check for CUDA (NVIDIA)
	if info := detectCUDA(); info != "" {
		return info
	}

	// Check for Metal (macOS)
	if runtime.GOOS == "darwin" {
		if info := detectMetal(); info != "" {
			return info
		}
	}

	// Check for Vulkan
	if info := detectVulkan(); info != "" {
		return info
	}

	return ""
}

// detectCUDA checks for NVIDIA CUDA support.
func detectCUDA() string {
	// Look up nvidia-smi in PATH for security
	nvidiaSmi, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return ""
	}

	cmd := exec.Command(nvidiaSmi, "--query-gpu=name", "--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil {
		return ""
	}

	name := strings.TrimSpace(string(out))
	lines := strings.Split(name, "\n")
	if len(lines) > 0 && lines[0] != "" {
		return fmt.Sprintf("CUDA: %s", lines[0])
	}
	return ""
}

// detectMetal checks for Apple Metal support.
func detectMetal() string {
	// Metal is available on macOS 10.11+
	// Look up system_profiler in PATH for security
	sysProfiler, err := exec.LookPath("system_profiler")
	if err != nil {
		// Fallback: assume Metal is available on macOS
		return "Metal: Apple GPU"
	}

	cmd := exec.Command(sysProfiler, "SPDisplaysDataType")
	out, err := cmd.Output()
	if err != nil {
		return "Metal: Apple GPU"
	}

	// Look for GPU chipset info
	lines := strings.Split(string(out), "\n")
	for _, line := range lines {
		if strings.Contains(line, "Chipset Model:") {
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				return fmt.Sprintf("Metal: %s", strings.TrimSpace(parts[1]))
			}
		}
	}

	// Default to Metal if on macOS
	return "Metal: Apple GPU"
}

// detectVulkan checks for Vulkan support.
func detectVulkan() string {
	// Look up vulkaninfo in PATH for security
	vulkanInfo, err := exec.LookPath("vulkaninfo")
	if err != nil {
		return ""
	}

	cmd := exec.Command(vulkanInfo, "--summary")
	out, err := cmd.Output()
	if err != nil {
		return ""
	}

	lines := strings.Split(string(out), "\n")
	for _, line := range lines {
		if strings.Contains(line, "deviceName") {
			parts := strings.SplitN(line, "=", 2)
			if len(parts) == 2 {
				return fmt.Sprintf("Vulkan: %s", strings.TrimSpace(parts[1]))
			}
		}
	}
	return ""
}

// ParseType parses a device type string.
func ParseType(s string) (Type, error) {
	switch strings.ToLower(s) {
	case "auto", "":
		return Auto, nil
	case "cpu":
		return CPU, nil
	case "gpu", "cuda", "metal", "vulkan":
		return GPU, nil
	default:
		return "", fmt.Errorf("unknown device type: %s (use: auto, cpu, gpu)", s)
	}
}

// SetEnvironment sets environment variables for the selected device.
func SetEnvironment(t Type) {
	switch t {
	case GPU:
		// Ensure GPU libraries are preferred
		os.Setenv("GGML_CUDA_ENABLE", "1")
	case CPU:
		// Force CPU-only mode
		os.Setenv("GGML_CUDA_ENABLE", "0")
	}
}
