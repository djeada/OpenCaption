package device

import (
	"testing"
)

func TestParseType(t *testing.T) {
	tests := []struct {
		input   string
		want    Type
		wantErr bool
	}{
		{"auto", Auto, false},
		{"AUTO", Auto, false},
		{"", Auto, false},
		{"cpu", CPU, false},
		{"CPU", CPU, false},
		{"gpu", GPU, false},
		{"GPU", GPU, false},
		{"cuda", GPU, false},
		{"metal", GPU, false},
		{"vulkan", GPU, false},
		{"invalid", "", true},
		{"tpu", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, err := ParseType(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Errorf("ParseType(%q) should have failed", tt.input)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseType(%q) failed: %v", tt.input, err)
			}
			if got != tt.want {
				t.Errorf("ParseType(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestDetect(t *testing.T) {
	// Test CPU detection (should always work)
	info := Detect(CPU)
	if info.Type != CPU {
		t.Errorf("Detect(CPU).Type = %q, want %q", info.Type, CPU)
	}
	if !info.Available {
		t.Error("CPU should always be available")
	}
	if info.Fallback {
		t.Error("CPU should not be a fallback")
	}

	// Test Auto detection
	info = Detect(Auto)
	if info.Type != CPU && info.Type != GPU {
		t.Errorf("Detect(Auto).Type = %q, want CPU or GPU", info.Type)
	}
	if !info.Available {
		t.Error("Auto detection should find an available device")
	}
}

func TestDetectGPU(t *testing.T) {
	// Test GPU detection - may or may not be available
	info := Detect(GPU)

	// If GPU is not available, it should fallback to CPU
	if !info.Available {
		if !info.Fallback {
			t.Error("GPU not available should set Fallback=true")
		}
		if info.FallbackTo != CPU {
			t.Errorf("FallbackTo = %q, want %q", info.FallbackTo, CPU)
		}
	}
}

func TestSetEnvironment(t *testing.T) {
	// Just ensure it doesn't panic
	SetEnvironment(CPU)
	SetEnvironment(GPU)
	SetEnvironment(Auto)
}
