.PHONY: build test assets clean

BINARY := cc-search
VERSION ?= dev
ONNX_LIB ?= $(shell \
	if [ -f /opt/homebrew/lib/libonnxruntime.dylib ]; then echo /opt/homebrew/lib/libonnxruntime.dylib; \
	elif [ -f /usr/local/lib/libonnxruntime.dylib ]; then echo /usr/local/lib/libonnxruntime.dylib; \
	elif [ -f /usr/local/lib/libonnxruntime.so ]; then echo /usr/local/lib/libonnxruntime.so; \
	else echo ""; fi)

build: internal/assets/model.onnx
	CGO_ENABLED=1 ONNX_LIB=$(ONNX_LIB) go build -ldflags="-X main.version=$(VERSION)" -o $(BINARY) ./cmd/cc-search

test: internal/assets/model.onnx
	CGO_ENABLED=1 ONNX_LIB=$(ONNX_LIB) go test ./... -v -count=1

assets: internal/assets/model.onnx

internal/assets/model.onnx:
	uv run python scripts/export-model.py

clean:
	rm -f $(BINARY)
