package embedding

import (
	"math"
	"testing"
)

func TestNewModel(t *testing.T) {
	model, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer model.Close()

	if model.ModelHash() == "" {
		t.Error("ModelHash should not be empty")
	}
}

func TestEmbedSingleText(t *testing.T) {
	model, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer model.Close()

	vecs, err := model.Embed([]string{"hello world"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 1 {
		t.Fatalf("got %d vectors, want 1", len(vecs))
	}
	if len(vecs[0]) != 384 {
		t.Fatalf("vector dim = %d, want 384", len(vecs[0]))
	}
}

func TestEmbedBatch(t *testing.T) {
	model, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer model.Close()

	texts := []string{"hello world", "how are you", "semantic search"}
	vecs, err := model.Embed(texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 3 {
		t.Fatalf("got %d vectors, want 3", len(vecs))
	}
	for i, v := range vecs {
		if len(v) != 384 {
			t.Errorf("vecs[%d] dim = %d, want 384", i, len(v))
		}
	}
}

func TestEmbedSimilarity(t *testing.T) {
	model, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer model.Close()

	vecs, err := model.Embed([]string{
		"How do I set up authentication?",
		"Configure JWT auth middleware",
		"The weather is nice today",
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	simAB := cosine(vecs[0], vecs[1])
	simAC := cosine(vecs[0], vecs[2])

	if simAB <= simAC {
		t.Errorf("auth questions should be more similar (%.4f) than auth vs weather (%.4f)", simAB, simAC)
	}
}

func TestEmbedDeterministic(t *testing.T) {
	model, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer model.Close()

	v1, _ := model.Embed([]string{"hello"})
	v2, _ := model.Embed([]string{"hello"})

	for i := range v1[0] {
		if v1[0][i] != v2[0][i] {
			t.Fatalf("embedding not deterministic at index %d: %f != %f", i, v1[0][i], v2[0][i])
		}
	}
}

func cosine(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
