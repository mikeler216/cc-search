package ingest

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
)

type turn struct {
	Role      string
	Text      string
	SessionID string
	Timestamp float64
	TurnIndex int
}

func extractText(message map[string]any) string {
	content, ok := message["content"]
	if !ok {
		return ""
	}

	switch c := content.(type) {
	case string:
		return strings.TrimSpace(c)
	case []any:
		var parts []string
		for _, block := range c {
			m, ok := block.(map[string]any)
			if !ok {
				continue
			}
			if m["type"] == "text" {
				if text, ok := m["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
		return strings.TrimSpace(strings.Join(parts, " "))
	}
	return ""
}

func projectNameFromPath(filePath string) string {
	parts := strings.Split(filePath, "/projects/")
	if len(parts) < 2 {
		return "unknown"
	}
	segments := strings.SplitN(parts[1], "/", 2)
	// Must have at least two segments: <project-name>/<file>
	if len(segments) < 2 || segments[0] == "" {
		return "unknown"
	}
	return segments[0]
}

func parseJSONLFile(filePath string) []turn {
	f, err := os.Open(filePath)
	if err != nil {
		return nil
	}
	defer f.Close()

	var turns []turn
	turnIndex := 0
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var entry map[string]any
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			continue
		}
		entryType, _ := entry["type"].(string)
		if entryType != "user" && entryType != "assistant" {
			continue
		}
		message, _ := entry["message"].(map[string]any)
		if message == nil {
			continue
		}
		text := extractText(message)
		if text == "" {
			continue
		}
		sessionID, _ := entry["sessionId"].(string)
		timestamp, _ := entry["timestamp"].(float64)

		turns = append(turns, turn{
			Role:      entryType,
			Text:      text,
			SessionID: sessionID,
			Timestamp: timestamp,
			TurnIndex: turnIndex,
		})
		turnIndex++
	}
	return turns
}
