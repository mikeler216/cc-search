package embedding

import (
	"os"
	"testing"
)

func loadTestVocab(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("../../internal/assets/vocab.txt")
	if err != nil {
		t.Fatalf("read vocab.txt: %v", err)
	}
	return data
}

func TestNewTokenizer(t *testing.T) {
	vocab := loadTestVocab(t)
	tok := NewTokenizer(vocab)
	if len(tok.vocab) == 0 {
		t.Fatal("vocab is empty")
	}
	if _, ok := tok.vocab["[CLS]"]; !ok {
		t.Error("missing [CLS] token")
	}
	if _, ok := tok.vocab["[SEP]"]; !ok {
		t.Error("missing [SEP] token")
	}
	if _, ok := tok.vocab["[UNK]"]; !ok {
		t.Error("missing [UNK] token")
	}
}

func TestTokenizeSimple(t *testing.T) {
	tok := NewTokenizer(loadTestVocab(t))
	ids, mask, typeIDs := tok.Tokenize("hello world")

	if len(ids) < 3 {
		t.Fatalf("expected at least 3 tokens ([CLS] + words + [SEP]), got %d", len(ids))
	}
	if ids[0] != tok.vocab["[CLS]"] {
		t.Errorf("first token should be [CLS], got id %d", ids[0])
	}
	if ids[len(ids)-1] != tok.vocab["[SEP]"] {
		t.Errorf("last token should be [SEP], got id %d", ids[len(ids)-1])
	}
	if len(mask) != len(ids) {
		t.Errorf("mask length %d != ids length %d", len(mask), len(ids))
	}
	for i, m := range mask {
		if m != 1 {
			t.Errorf("mask[%d] = %d, want 1", i, m)
		}
	}
	if len(typeIDs) != len(ids) {
		t.Errorf("typeIDs length %d != ids length %d", len(typeIDs), len(ids))
	}
}

func TestCountTokens(t *testing.T) {
	tok := NewTokenizer(loadTestVocab(t))

	count := tok.CountTokens("hello world")
	if count < 2 {
		t.Errorf("expected at least 2 tokens for 'hello world', got %d", count)
	}

	count = tok.CountTokens("")
	if count != 0 {
		t.Errorf("expected 0 tokens for empty string, got %d", count)
	}
}

func TestTokenizeHandlesPunctuation(t *testing.T) {
	tok := NewTokenizer(loadTestVocab(t))
	ids, _, _ := tok.Tokenize("Hello, world!")
	if len(ids) < 5 {
		t.Errorf("expected at least 5 tokens (CLS + hello + , + world + ! + SEP), got %d", len(ids))
	}
}

func TestTokenizeUnknownWord(t *testing.T) {
	tok := NewTokenizer(loadTestVocab(t))
	// Use a word containing a rare Unicode Lo character (ꗉ U+A5C9) that has
	// no entry in the BERT vocab, so WordPiece cannot segment it and falls
	// back to [UNK].  Pure-ASCII "nonsense" words like "xyzzyplugh" are always
	// segmentable letter-by-letter because the vocab contains every ##a-##z.
	ids, _, _ := tok.Tokenize("ꗉꗉꗉ")
	unkID := tok.vocab["[UNK]"]
	hasUnk := false
	for _, id := range ids {
		if id == unkID {
			hasUnk = true
			break
		}
	}
	if !hasUnk {
		t.Error("expected [UNK] token for word with characters absent from vocab")
	}
}
