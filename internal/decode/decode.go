package decode

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"os/exec"
)

// DecodeToPCM16 uses ffmpeg to decode input to mono 16k PCM float32 samples.
func DecodeToPCM16(in, ffmpegPath string) ([]float32, error) {
	if ffmpegPath == "" {
		ffmpegPath = "ffmpeg"
	}
	if _, err := exec.LookPath(ffmpegPath); err != nil {
		return nil, fmt.Errorf("ffmpeg not found (set --ffmpeg-path or add to PATH): %w", err)
	}

	input := in
	if in == "-" {
		input = "pipe:0"
	}
	cmd := exec.Command(ffmpegPath,
		"-hide_banner",
		"-loglevel", "error",
		"-i", input,
		"-ac", "1",
		"-ar", "16000",
		"-f", "s16le",
		"-acodec", "pcm_s16le",
		"pipe:1",
	)
	if in == "-" {
		cmd.Stdin = os.Stdin
	}

	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}

	data, err := io.ReadAll(stdout)
	if err != nil {
		_ = cmd.Wait()
		return nil, err
	}
	if err := cmd.Wait(); err != nil {
		return nil, fmt.Errorf("ffmpeg: %v\n%s", err, stderr.String())
	}
	if len(data) < 2 {
		return nil, fmt.Errorf("ffmpeg produced no PCM output")
	}
	if len(data)%2 != 0 {
		data = data[:len(data)-1]
	}

	samples := make([]float32, len(data)/2)
	for i := 0; i < len(data); i += 2 {
		v := int16(binary.LittleEndian.Uint16(data[i : i+2]))
		samples[i/2] = float32(v) / 32768.0
	}
	return samples, nil
}
