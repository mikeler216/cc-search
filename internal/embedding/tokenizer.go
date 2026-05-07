package embedding

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// Tokenizer implements a BERT-style WordPiece tokenizer.
// It lowercases input, strips accents, splits on whitespace and punctuation,
// then applies greedy longest-match subword segmentation with a ## prefix for
// continuation pieces. Unknown words that cannot be segmented become [UNK].
type Tokenizer struct {
	vocab  map[string]int64
	maxLen int
}

// NewTokenizer loads a WordPiece vocabulary from the contents of vocab.txt.
// Each non-empty line's zero-based index is its token ID. Interior blank
// lines are skipped but not accounted for — the vocab file must not contain
// interior blank lines (BERT vocab files satisfy this).
func NewTokenizer(vocabData []byte) *Tokenizer {
	vocab := make(map[string]int64)
	for i, line := range strings.Split(string(vocabData), "\n") {
		line = strings.TrimRight(line, "\r")
		if line != "" {
			vocab[line] = int64(i)
		}
	}
	return &Tokenizer{vocab: vocab, maxLen: 512}
}

// Tokenize converts text into three parallel int64 slices suitable for BERT
// model input: token IDs, attention mask (all 1s), and token type IDs (all 0s).
// The sequence is wrapped with [CLS] at the start and [SEP] at the end and
// truncated to maxLen tokens total when necessary.
func (t *Tokenizer) Tokenize(text string) (inputIDs, attentionMask, tokenTypeIDs []int64) {
	tokens := t.tokenizeText(text)
	if len(tokens) > t.maxLen-2 {
		tokens = tokens[:t.maxLen-2]
	}

	ids := make([]int64, 0, len(tokens)+2)
	ids = append(ids, t.vocab["[CLS]"])
	for _, tok := range tokens {
		ids = append(ids, t.vocab[tok])
	}
	ids = append(ids, t.vocab["[SEP]"])

	mask := make([]int64, len(ids))
	typeIDs := make([]int64, len(ids))
	for i := range mask {
		mask[i] = 1
	}
	return ids, mask, typeIDs
}

// CountTokens returns the number of WordPiece tokens in text, before any
// truncation. This may exceed maxLen-2; use it to decide whether content
// needs to be split into smaller chunks.
func (t *Tokenizer) CountTokens(text string) int {
	return len(t.tokenizeText(text))
}

// tokenizeText applies the full BERT pre-processing pipeline:
// lowercase -> accent stripping -> whitespace/punctuation splitting -> WordPiece.
func (t *Tokenizer) tokenizeText(text string) []string {
	text = strings.ToLower(text)
	text = stripAccents(text)
	words := splitOnWhitespaceAndPunctuation(text)

	var tokens []string
	for _, word := range words {
		tokens = append(tokens, t.wordPiece(word)...)
	}
	return tokens
}

// stripAccents NFD-normalizes s and drops all Unicode non-spacing marks (Mn
// category), which effectively removes diacritics from base characters.
func stripAccents(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range norm.NFD.String(s) {
		if !unicode.Is(unicode.Mn, r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// splitOnWhitespaceAndPunctuation splits text into tokens by treating every
// whitespace run as a delimiter and every punctuation/symbol character as both
// a delimiter and a standalone token. Control characters are silently dropped.
func splitOnWhitespaceAndPunctuation(text string) []string {
	var tokens []string
	var current strings.Builder
	for _, r := range text {
		if unicode.IsSpace(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
		} else if isPunct(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			tokens = append(tokens, string(r))
		} else if unicode.IsControl(r) {
			continue
		} else {
			current.WriteRune(r)
		}
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}

// isPunct reports whether r is an ASCII punctuation character or a Unicode
// punctuation rune (P* category). This matches BERT's _is_punctuation exactly.
// Unicode Symbol characters (S* category, e.g. ©, ™, °, £, €) are intentionally
// excluded — BERT does not treat them as word boundaries.
func isPunct(r rune) bool {
	if (r >= '!' && r <= '/') || (r >= ':' && r <= '@') ||
		(r >= '[' && r <= '`') || (r >= '{' && r <= '~') {
		return true
	}
	return unicode.IsPunct(r)
}

// wordPiece segments word into one or more subword tokens using greedy
// longest-match. If the whole word is in the vocabulary it is returned as-is.
// Continuation pieces are prefixed with "##". If no valid segmentation exists
// the whole word is replaced with "[UNK]".
func (t *Tokenizer) wordPiece(word string) []string {
	if _, ok := t.vocab[word]; ok {
		return []string{word}
	}

	runes := []rune(word)
	var tokens []string
	start := 0

	for start < len(runes) {
		end := len(runes)
		var matched string
		for end > start {
			substr := string(runes[start:end])
			if start > 0 {
				substr = "##" + substr
			}
			if _, ok := t.vocab[substr]; ok {
				matched = substr
				break
			}
			end--
		}
		if matched == "" {
			return []string{"[UNK]"}
		}
		tokens = append(tokens, matched)
		start = end
	}
	return tokens
}
