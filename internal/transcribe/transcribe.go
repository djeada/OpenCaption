package transcribe

import (
	"errors"
	"io"
	"strings"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

// Transcribe runs whisper.cpp over PCM audio and returns segments.
func Transcribe(ctx whisper.Context, pcm []float32, language string) ([]whisper.Segment, error) {
	language = strings.ToLower(strings.TrimSpace(language))
	if language == "" {
		language = "auto"
	}
	if err := ctx.SetLanguage(language); err != nil {
		// English-only models already default to English. The binding rejects
		// every SetLanguage call for them, including "en" and "auto".
		if !errors.Is(err, whisper.ErrModelNotMultilingual) || (language != "auto" && language != "en") {
			return nil, err
		}
	}
	if err := ctx.Process(pcm, nil, nil, nil); err != nil {
		return nil, err
	}
	var segs []whisper.Segment
	for {
		seg, err := ctx.NextSegment()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
		txt := strings.TrimSpace(seg.Text)
		if txt == "" {
			continue
		}
		seg.Text = txt
		segs = append(segs, seg)
	}
	return segs, nil
}
