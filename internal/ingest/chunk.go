package ingest

import "strings"

type tokenCounter interface {
	CountTokens(text string) int
}

func chunkText(text string, counter tokenCounter, maxTokens, overlapTokens int) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	if counter.CountTokens(text) <= maxTokens {
		return []string{text}
	}

	words := strings.Fields(text)
	var chunks []string
	start := 0

	for start < len(words) {
		end := start + maxTokens
		if end > len(words) {
			end = len(words)
		}

		for end > start+1 {
			candidate := strings.Join(words[start:end], " ")
			if counter.CountTokens(candidate) <= maxTokens {
				break
			}
			end--
		}

		chunk := strings.Join(words[start:end], " ")
		chunks = append(chunks, chunk)

		if end >= len(words) {
			break
		}
		step := end - start - overlapTokens
		if step < 1 {
			step = 1
		}
		start += step
	}

	return chunks
}
