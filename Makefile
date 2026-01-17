SHELL := /bin/sh
BINARY := opencaption
CMD := ./cmd/opencaption
PREFIX ?= $(HOME)/.local
BIN_DIR ?= $(PREFIX)/bin
LIB_DIR ?= $(PREFIX)/lib/opencaption
WHISPER_CPP ?= $(HOME)/.local/src/whisper.cpp
RPATH ?= 1
MODEL ?= base.en
WHISPER_INC := $(WHISPER_CPP)/include
GGML_INC := $(WHISPER_CPP)/ggml/include
WHISPER_LIB := $(WHISPER_CPP)/build/src
GGML_LIB := $(WHISPER_CPP)/build/ggml/src

export CGO_CFLAGS := -I$(WHISPER_INC) -I$(GGML_INC)
CGO_LDFLAGS := -L$(WHISPER_LIB) -L$(GGML_LIB)
ifeq ($(RPATH),1)
CGO_LDFLAGS += -Wl,-rpath,$(WHISPER_LIB) -Wl,-rpath,$(GGML_LIB) -Wl,-rpath,$(LIB_DIR)
endif
export CGO_LDFLAGS

.PHONY: all setup check build run install install-libs uninstall test lint fmt clean help

all: build

setup:
	@bash ./scripts/setup_model.sh --model "$(MODEL)" --prefix "$(WHISPER_CPP)"

check:
	@test -f "$(WHISPER_INC)/whisper.h" || (echo "missing whisper.cpp headers at $(WHISPER_INC)"; echo "run: make setup (MODEL=$(MODEL)) or set WHISPER_CPP=/path/to/whisper.cpp"; exit 1)
	@test -f "$(WHISPER_LIB)/libwhisper.so" || (echo "missing whisper.cpp libs at $(WHISPER_LIB)"; echo "run: make setup (MODEL=$(MODEL)) or set WHISPER_CPP=/path/to/whisper.cpp"; exit 1)
	@test -f "$(GGML_LIB)/libggml.so" || (echo "missing ggml libs at $(GGML_LIB)"; echo "run: make setup (MODEL=$(MODEL)) or set WHISPER_CPP=/path/to/whisper.cpp"; exit 1)

build: check
	go build -o $(BINARY) $(CMD)

run: build
	@LD_LIBRARY_PATH="$(WHISPER_LIB):$(GGML_LIB):$$LD_LIBRARY_PATH" ./$(BINARY) $(RUN_ARGS)

install-libs: check
	@mkdir -p "$(LIB_DIR)"
	@cp -a "$(WHISPER_LIB)"/libwhisper.so* "$(LIB_DIR)/"
	@cp -a "$(GGML_LIB)"/libggml*.so* "$(LIB_DIR)/"
	@echo "installed libs to $(LIB_DIR)"

install: build install-libs
	@mkdir -p "$(BIN_DIR)"
	@cp "$(BINARY)" "$(BIN_DIR)/$(BINARY)"
	@echo "installed to $(BIN_DIR)/$(BINARY)"

uninstall:
	@rm -f "$(BIN_DIR)/$(BINARY)"
	@echo "removed $(BIN_DIR)/$(BINARY)"

test:
	go test ./...

lint:
	go vet ./...

fmt:
	gofmt -w cmd internal

clean:
	rm -f $(BINARY)

help:
	@echo "Targets:"
	@echo "  build        Build the opencaption binary"
	@echo "  run          Build and run (use RUN_ARGS=\"...\")"
	@echo "  setup        Install/build whisper.cpp and download a model"
	@echo "  install      Install to \$$BIN_DIR (default $(BIN_DIR))"
	@echo "  install-libs Install shared libs to \$$LIB_DIR (default $(LIB_DIR))"
	@echo "  uninstall    Remove installed binary"
	@echo "  test         Run tests"
	@echo "  lint         Run go vet"
	@echo "  fmt          Format Go files"
	@echo "  clean        Remove the local binary"
	@echo "Vars:"
	@echo "  WHISPER_CPP  Path to whisper.cpp checkout (default $(WHISPER_CPP))"
	@echo "  MODEL        Model name for setup (default $(MODEL))"
	@echo "  RPATH        Embed runtime library paths (default 1)"
	@echo "  LIB_DIR      Install path for shared libs (default $(LIB_DIR))"
