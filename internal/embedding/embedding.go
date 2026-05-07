// Package embedding wraps an ONNX sentence-transformers model
// (all-MiniLM-L6-v2) and the matching WordPiece tokenizer to produce
// 384-dimensional sentence embeddings.
//
// Pipeline per call:
//  1. Tokenize each input string into (input_ids, attention_mask, token_type_ids).
//  2. Pad each batch to its longest sequence (token id 0, mask 0).
//  3. Run ONNX inference, yielding [batch, seq_len, 384] hidden states.
//  4. Mean-pool over the sequence dimension, weighted by the attention mask.
//  5. L2-normalize the resulting 384-dim vectors so cosine similarity == dot product.
package embedding

import (
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/mikeler216/cc-search/internal/assets"
)

// defaultBatchSize bounds peak memory by capping how many texts feed into a
// single ONNX inference call.
const defaultBatchSize = 64

// ortInitOnce guards the one-time global ONNX Runtime initialization.
// The runtime panics if InitializeEnvironment is called twice, so we use
// sync.Once to make concurrent New() calls safe.
var (
	ortInitOnce sync.Once
	ortInitErr  error
)

// Model is a thread-unsafe wrapper around the embedded ONNX sentence
// transformer. A single DynamicAdvancedSession is created in New() and
// reused across all batches for the lifetime of the Model.
type Model struct {
	modelPath string
	tokenizer *Tokenizer
	modelHash string
	session   *ort.DynamicAdvancedSession
}

// onnxLibPath returns the conventional install path for the ONNX Runtime
// shared library on the current platform. Windows is unsupported by
// default; users on Windows must set ONNX_LIB explicitly.
func onnxLibPath() string {
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			return "/opt/homebrew/lib/libonnxruntime.dylib"
		}
		return "/usr/local/lib/libonnxruntime.dylib"
	case "windows":
		return "" // unsupported on Windows; user must set ONNX_LIB
	default:
		return "/usr/local/lib/libonnxruntime.so"
	}
}

// New initializes the global ONNX Runtime (once per process), writes the
// embedded model bytes to a temp file (so onnxruntime can mmap them),
// creates a single dynamic session for the lifetime of the Model, and
// returns it ready for inference. The caller must Close the returned
// Model to destroy the session and remove the temp directory.
//
// The ONNX_LIB environment variable, if set, overrides the auto-detected
// shared library path.
func New() (*Model, error) {
	ortInitOnce.Do(func() {
		libPath := onnxLibPath()
		if p := os.Getenv("ONNX_LIB"); p != "" {
			libPath = p
		}
		ort.SetSharedLibraryPath(libPath)
		ortInitErr = ort.InitializeEnvironment()
	})
	if ortInitErr != nil {
		return nil, fmt.Errorf("init onnxruntime: %w", ortInitErr)
	}

	tmpDir, err := os.MkdirTemp("", "cc-search-model-*")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}
	modelPath := filepath.Join(tmpDir, "model.onnx")
	if err := os.WriteFile(modelPath, assets.ModelONNX, 0o644); err != nil {
		os.RemoveAll(tmpDir)
		return nil, fmt.Errorf("write model file: %w", err)
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		nil,
	)
	if err != nil {
		os.RemoveAll(tmpDir)
		return nil, fmt.Errorf("create session: %w", err)
	}

	hash := sha256.Sum256(assets.ModelONNX)

	return &Model{
		modelPath: modelPath,
		tokenizer: NewTokenizer(assets.VocabTxt),
		modelHash: fmt.Sprintf("%x", hash),
		session:   session,
	}, nil
}

// Close destroys the ONNX session and removes the temp directory holding
// the on-disk model file. The global ONNX Runtime environment is
// intentionally left initialized so a subsequent New() call in the same
// process is fast.
func (m *Model) Close() {
	if m == nil {
		return
	}
	if m.session != nil {
		_ = m.session.Destroy()
		m.session = nil
	}
	if m.modelPath != "" {
		os.RemoveAll(filepath.Dir(m.modelPath))
	}
}

// ModelHash returns the SHA256 of the embedded model bytes, hex-encoded.
// Used to invalidate the vector store when the model changes.
func (m *Model) ModelHash() string {
	return m.modelHash
}

// CountTokens returns the WordPiece token count for text, before any
// truncation or special-token wrapping. Useful for chunking decisions.
func (m *Model) CountTokens(text string) int {
	return m.tokenizer.CountTokens(text)
}

// Embed returns 384-dimensional embedding vectors for each input text.
// Inputs longer than 510 WordPiece tokens are silently truncated. Callers
// who need to keep all content should split text into chunks using
// CountTokens to check length first.
func (m *Model) Embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	results := make([][]float32, len(texts))
	for start := 0; start < len(texts); start += defaultBatchSize {
		end := start + defaultBatchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[start:end]
		vecs, err := m.embedBatch(batch)
		if err != nil {
			return nil, fmt.Errorf("embed batch [%d:%d]: %w", start, end, err)
		}
		copy(results[start:end], vecs)
	}
	return results, nil
}

// embedBatch tokenizes a single batch, runs ONNX inference on the shared
// session, and applies attention-mask-weighted mean pooling followed by
// L2 normalization.
//
// All sequences in the batch are padded to the longest sequence (not to
// maxLen=512) — padding tokens get id=0 and mask=0, which the mean-pool
// step naturally ignores.
func (m *Model) embedBatch(texts []string) ([][]float32, error) {
	batchSize := int64(len(texts))

	// Tokenize and find the longest sequence for padding.
	allIDs := make([][]int64, batchSize)
	allMasks := make([][]int64, batchSize)
	allTypes := make([][]int64, batchSize)
	maxLen := int64(0)

	for i, text := range texts {
		ids, mask, typeIDs := m.tokenizer.Tokenize(text)
		allIDs[i] = ids
		allMasks[i] = mask
		allTypes[i] = typeIDs
		if int64(len(ids)) > maxLen {
			maxLen = int64(len(ids))
		}
	}
	if maxLen == 0 {
		// All inputs tokenized to empty — produce zero vectors. In practice
		// the tokenizer always emits at least [CLS][SEP], so this is defensive.
		zero := make([][]float32, batchSize)
		for i := range zero {
			zero[i] = make([]float32, 384)
		}
		return zero, nil
	}

	// Flatten ragged token sequences into row-major dense [batch, maxLen] arrays.
	flatIDs := make([]int64, batchSize*maxLen)
	flatMask := make([]int64, batchSize*maxLen)
	flatTypes := make([]int64, batchSize*maxLen)

	for i := int64(0); i < batchSize; i++ {
		seqLen := int64(len(allIDs[i]))
		for j := int64(0); j < seqLen; j++ {
			flatIDs[i*maxLen+j] = allIDs[i][j]
			flatMask[i*maxLen+j] = allMasks[i][j]
			flatTypes[i*maxLen+j] = allTypes[i][j]
		}
		// Padding positions remain zero in all three arrays.
	}

	shape := ort.NewShape(batchSize, maxLen)
	inputIDs, err := ort.NewTensor(shape, flatIDs)
	if err != nil {
		return nil, fmt.Errorf("create input_ids tensor: %w", err)
	}
	defer inputIDs.Destroy()

	attentionMask, err := ort.NewTensor(shape, flatMask)
	if err != nil {
		return nil, fmt.Errorf("create attention_mask tensor: %w", err)
	}
	defer attentionMask.Destroy()

	tokenTypeIDs, err := ort.NewTensor(shape, flatTypes)
	if err != nil {
		return nil, fmt.Errorf("create token_type_ids tensor: %w", err)
	}
	defer tokenTypeIDs.Destroy()

	// Output is [batch, seq_len, 384] last_hidden_state from the encoder.
	outputShape := ort.NewShape(batchSize, maxLen, 384)
	output, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer output.Destroy()

	if err := m.session.Run(
		[]ort.Value{inputIDs, attentionMask, tokenTypeIDs},
		[]ort.Value{output},
	); err != nil {
		return nil, fmt.Errorf("run session: %w", err)
	}

	// Mean-pool weighted by attention mask, then L2-normalize.
	// raw is laid out as [batch * seq_len * 384] in C row-major order.
	raw := output.GetData()
	results := make([][]float32, batchSize)
	const dim = int64(384)

	for i := int64(0); i < batchSize; i++ {
		seqLen := int64(len(allIDs[i]))
		vec := make([]float32, dim)
		maskSum := float32(0)

		for j := int64(0); j < seqLen; j++ {
			maskVal := float32(flatMask[i*maxLen+j])
			if maskVal == 0 {
				continue
			}
			maskSum += maskVal
			rowOffset := i*maxLen*dim + j*dim
			for k := int64(0); k < dim; k++ {
				vec[k] += raw[rowOffset+k] * maskVal
			}
		}

		if maskSum > 0 {
			inv := 1.0 / maskSum
			for k := range vec {
				vec[k] *= inv
			}
		}

		// L2 normalize so cosine similarity reduces to a dot product.
		var norm float32
		for _, v := range vec {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm > 0 {
			inv := 1.0 / norm
			for k := range vec {
				vec[k] *= inv
			}
		}

		results[i] = vec
	}

	return results, nil
}
