package vad

import (
	"math"
	"testing"
)

func TestSpeechSegmentsEmptyInput(t *testing.T) {
	segs := SpeechSegments(nil, 16000)
	if segs != nil {
		t.Errorf("SpeechSegments(nil) should return nil, got %v", segs)
	}

	segs = SpeechSegments([]float32{}, 16000)
	if segs != nil {
		t.Errorf("SpeechSegments([]) should return nil, got %v", segs)
	}
}

func TestSpeechSegmentsInvalidSampleRate(t *testing.T) {
	pcm := make([]float32, 1000)
	segs := SpeechSegments(pcm, 0)
	if segs != nil {
		t.Errorf("SpeechSegments with sr=0 should return nil, got %v", segs)
	}

	segs = SpeechSegments(pcm, -1)
	if segs != nil {
		t.Errorf("SpeechSegments with sr=-1 should return nil, got %v", segs)
	}
}

func TestSpeechSegmentsShortAudio(t *testing.T) {
	sr := 16000
	// Less than half a second
	pcm := make([]float32, sr/4)
	for i := range pcm {
		pcm[i] = 0.5
	}

	segs := SpeechSegments(pcm, sr)
	if len(segs) != 1 {
		t.Fatalf("Expected 1 segment for short audio, got %d", len(segs))
	}
	if segs[0].Start != 0 || segs[0].End != len(pcm) {
		t.Errorf("Short audio should return full range, got Start=%d End=%d", segs[0].Start, segs[0].End)
	}
}

func TestSpeechSegmentsSilence(t *testing.T) {
	sr := 16000
	// 2 seconds of silence
	pcm := make([]float32, sr*2)
	// All zeros = silence

	segs := SpeechSegments(pcm, sr)
	// Should return full range if no speech detected
	if len(segs) != 1 {
		t.Fatalf("Expected 1 segment for silence, got %d", len(segs))
	}
}

func TestSpeechSegmentsWithSpeech(t *testing.T) {
	sr := 16000
	// 3 seconds of audio
	pcm := make([]float32, sr*3)

	// Add speech in the middle (1-2 seconds)
	for i := sr; i < sr*2; i++ {
		// Simulate speech with varying amplitude
		pcm[i] = 0.3 * float32(math.Sin(float64(i)*0.1))
	}

	segs := SpeechSegments(pcm, sr)
	if len(segs) == 0 {
		t.Fatal("Expected at least 1 segment for audio with speech")
	}

	// The speech segment should be roughly in the middle
	found := false
	for _, seg := range segs {
		if seg.Start < sr*2 && seg.End > sr {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Expected to find speech segment around 1-2 seconds, got %v", segs)
	}
}

func TestFrameRMS(t *testing.T) {
	// Test with known values
	frameLen := 100
	pcm := make([]float32, 300)

	// First frame: all 0.5 -> RMS = 0.5
	for i := 0; i < 100; i++ {
		pcm[i] = 0.5
	}
	// Second frame: all 0 -> RMS = 0
	// Third frame: all 1.0 -> RMS = 1.0
	for i := 200; i < 300; i++ {
		pcm[i] = 1.0
	}

	rms := frameRMS(pcm, frameLen)
	if len(rms) != 3 {
		t.Fatalf("Expected 3 RMS values, got %d", len(rms))
	}

	// Check approximate values
	if math.Abs(float64(rms[0]-0.5)) > 0.01 {
		t.Errorf("RMS[0] = %f, want ~0.5", rms[0])
	}
	if math.Abs(float64(rms[1])) > 0.01 {
		t.Errorf("RMS[1] = %f, want ~0", rms[1])
	}
	if math.Abs(float64(rms[2]-1.0)) > 0.01 {
		t.Errorf("RMS[2] = %f, want ~1.0", rms[2])
	}
}

func TestFrameRMSEmpty(t *testing.T) {
	rms := frameRMS(nil, 100)
	if rms != nil {
		t.Errorf("frameRMS(nil) should return nil, got %v", rms)
	}

	rms = frameRMS([]float32{}, 100)
	if rms != nil {
		t.Errorf("frameRMS([]) should return nil, got %v", rms)
	}
}

func TestFrameRMSTooShort(t *testing.T) {
	pcm := make([]float32, 50)
	rms := frameRMS(pcm, 100)
	if rms != nil {
		t.Errorf("frameRMS with short pcm should return nil, got %v", rms)
	}
}

func TestEnergyThreshold(t *testing.T) {
	// Create RMS values with known distribution
	rms := []float32{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10}

	threshold := energyThreshold(rms)

	// Threshold should be above the noise floor but reasonable
	if threshold < 0.01 {
		t.Errorf("Threshold = %f, should be at least 0.01", threshold)
	}
	if threshold > 1.0 {
		t.Errorf("Threshold = %f, should be less than 1.0", threshold)
	}
}

func TestFramesToSegment(t *testing.T) {
	tests := []struct {
		name         string
		startFrame   int
		endFrame     int
		padFrames    int
		frameLen     int
		totalSamples int
		wantStart    int
		wantEnd      int
	}{
		{
			name:         "basic segment",
			startFrame:   10,
			endFrame:     20,
			padFrames:    2,
			frameLen:     100,
			totalSamples: 3000,
			wantStart:    800,  // (10-2)*100
			wantEnd:      2300, // (20+1+2)*100
		},
		{
			name:         "clamp to start",
			startFrame:   1,
			endFrame:     5,
			padFrames:    5,
			frameLen:     100,
			totalSamples: 1000,
			wantStart:    0, // clamped
			wantEnd:      1000,
		},
		{
			name:         "clamp to end",
			startFrame:   8,
			endFrame:     10,
			padFrames:    2,
			frameLen:     100,
			totalSamples: 1000,
			wantStart:    600,
			wantEnd:      1000, // clamped
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			seg := framesToSegment(tt.startFrame, tt.endFrame, tt.padFrames, tt.frameLen, tt.totalSamples)
			if seg.Start != tt.wantStart {
				t.Errorf("Start = %d, want %d", seg.Start, tt.wantStart)
			}
			if seg.End != tt.wantEnd {
				t.Errorf("End = %d, want %d", seg.End, tt.wantEnd)
			}
		})
	}
}
