package transcribe

import (
	"errors"
	"io"
	"testing"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

type languageContext struct {
	whisper.Context
	languageErr error
	language    string
	processed   bool
}

func (c *languageContext) SetLanguage(language string) error {
	c.language = language
	return c.languageErr
}

func (c *languageContext) Process([]float32, whisper.EncoderBeginCallback, whisper.SegmentCallback, whisper.ProgressCallback) error {
	c.processed = true
	return nil
}

func (c *languageContext) NextSegment() (whisper.Segment, error) {
	return whisper.Segment{}, io.EOF
}

func TestTranscribeLanguageSelection(t *testing.T) {
	otherError := errors.New("unexpected language error")
	tests := []struct {
		name, language, wantLanguage string
		languageErr, wantErr         error
	}{
		{"English-only default", "", "auto", whisper.ErrModelNotMultilingual, nil},
		{"English-only auto", "auto", "auto", whisper.ErrModelNotMultilingual, nil},
		{"English-only explicit English", "en", "en", whisper.ErrModelNotMultilingual, nil},
		{"normalized language", " EN ", "en", whisper.ErrModelNotMultilingual, nil},
		{"English-only rejects other language", "fr", "fr", whisper.ErrModelNotMultilingual, whisper.ErrModelNotMultilingual},
		{"multilingual default", "", "auto", nil, nil},
		{"multilingual explicit language", "fr", "fr", nil, nil},
		{"other errors propagate", "en", "en", otherError, otherError},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &languageContext{languageErr: tt.languageErr}
			_, err := Transcribe(ctx, []float32{0}, tt.language)
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("Transcribe() = %v, want %v", err, tt.wantErr)
			}
			if ctx.processed != (tt.wantErr == nil) || ctx.language != tt.wantLanguage {
				t.Errorf("unexpected language handling: %+v", ctx)
			}
		})
	}
}
