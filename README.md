# OpenCaption


go build -o captioner .
# Example:
./captioner -in "talk.mp4" \
  -out "talk.vtt" \
  -model "whisper.cpp/models/ggml-base.en.bin" \
  -lang en \
  -window 60 -overlap 1 \
  -format vtt
