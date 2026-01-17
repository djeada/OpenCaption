package vad

import (
	"math"
	"sort"
)

// Segment represents a speech window in sample indices [Start, End).
type Segment struct {
	Start int
	End   int
}

// SpeechSegments returns speech-only segments using simple energy-based VAD.
func SpeechSegments(pcm []float32, sr int) []Segment {
	if len(pcm) == 0 || sr <= 0 {
		return nil
	}
	// For short audio, avoid aggressive slicing.
	if len(pcm) < sr/2 {
		return []Segment{{Start: 0, End: len(pcm)}}
	}

	frameLen := int(0.03 * float64(sr)) // 30 ms frames
	if frameLen <= 0 {
		return []Segment{{Start: 0, End: len(pcm)}}
	}
	rms := frameRMS(pcm, frameLen)
	if len(rms) == 0 {
		return []Segment{{Start: 0, End: len(pcm)}}
	}

	threshold := energyThreshold(rms)
	segs := speechRuns(rms, threshold, frameLen, len(pcm), sr)
	if len(segs) == 0 {
		return []Segment{{Start: 0, End: len(pcm)}}
	}
	return segs
}

func frameRMS(pcm []float32, frameLen int) []float32 {
	count := len(pcm) / frameLen
	if count == 0 {
		return nil
	}
	rms := make([]float32, count)
	for i := 0; i < count; i++ {
		start := i * frameLen
		end := start + frameLen
		var sum float64
		for _, v := range pcm[start:end] {
			sum += float64(v * v)
		}
		rms[i] = float32(math.Sqrt(sum / float64(frameLen)))
	}
	return rms
}

func energyThreshold(rms []float32) float32 {
	tmp := make([]float32, len(rms))
	copy(tmp, rms)
	sort.Slice(tmp, func(i, j int) bool { return tmp[i] < tmp[j] })

	p := len(tmp) / 5 // 20th percentile
	if p < 0 {
		p = 0
	}
	noise := tmp[p]
	thr := noise * 3.0
	if thr < 0.01 {
		thr = 0.01
	}
	return thr
}

func speechRuns(rms []float32, threshold float32, frameLen int, totalSamples int, sr int) []Segment {
	const (
		padSec    = 0.2
		gapSec    = 0.25
		minSegSec = 0.3
	)
	frameSec := float64(frameLen) / float64(sr)
	padFrames := int(padSec / frameSec)
	if padFrames < 1 {
		padFrames = 1
	}
	gapFrames := int(gapSec / frameSec)
	if gapFrames < 1 {
		gapFrames = 1
	}
	minSegFrames := int(minSegSec / frameSec)
	if minSegFrames < 1 {
		minSegFrames = 1
	}

	var segs []Segment
	inSpeech := false
	startFrame := 0
	lastSpeech := 0
	silenceCount := 0

	for i, v := range rms {
		if v >= threshold {
			if !inSpeech {
				inSpeech = true
				startFrame = i
			}
			lastSpeech = i
			silenceCount = 0
			continue
		}
		if inSpeech {
			silenceCount++
			if silenceCount >= gapFrames {
				segs = append(segs, framesToSegment(startFrame, lastSpeech, padFrames, frameLen, totalSamples))
				inSpeech = false
				silenceCount = 0
			}
		}
	}
	if inSpeech {
		segs = append(segs, framesToSegment(startFrame, lastSpeech, padFrames, frameLen, totalSamples))
	}

	var out []Segment
	for _, s := range segs {
		if s.End-s.Start < minSegFrames*frameLen {
			continue
		}
		out = append(out, s)
	}
	return out
}

func framesToSegment(startFrame, endFrame, padFrames, frameLen, totalSamples int) Segment {
	start := (startFrame - padFrames) * frameLen
	if start < 0 {
		start = 0
	}
	end := (endFrame + 1 + padFrames) * frameLen
	if end > totalSamples {
		end = totalSamples
	}
	if end < start {
		end = start
	}
	return Segment{Start: start, End: end}
}
