package ingest

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/mikeler216/cc-search/internal/embedding"
	"github.com/mikeler216/cc-search/internal/store"
)

// Run is the top-level ingest entry point. It opens (or creates) the vector
// store at dbPath, performs a version check, discovers all JSONL conversation
// files under claudeDir/projects, deletes records for files that no longer
// exist, and indexes any file whose mtime has changed (or all files when
// full=true). It returns the total number of chunks and files indexed in
// this run.
func Run(dbPath string, model *embedding.Model, claudeDir string, full bool) (indexedChunks, indexedFiles int, err error) {
	db, err := store.Open(dbPath)
	if err != nil {
		return 0, 0, fmt.Errorf("open store: %w", err)
	}
	defer db.Close()

	// Version check: if model or schema changed, drop everything and reindex.
	match, err := db.CheckVersion(model.ModelHash())
	if err != nil || !match {
		if dropErr := db.DropAll(); dropErr != nil {
			return 0, 0, fmt.Errorf("drop all: %w", dropErr)
		}
		full = true
	}
	if err := db.InitMeta(model.ModelHash()); err != nil {
		return 0, 0, fmt.Errorf("init meta: %w", err)
	}

	// Discover all JSONL files on disk.
	files := findJSONLFiles(claudeDir)
	fileSet := make(map[string]bool, len(files))
	for _, f := range files {
		fileSet[f] = true
	}

	// Get all files currently tracked in the DB.
	existing, err := db.AllFiles()
	if err != nil {
		return 0, 0, fmt.Errorf("list files: %w", err)
	}

	// Delete chunks for files that no longer exist on disk.
	for path := range existing {
		if !fileSet[path] {
			// Ignore errors from deletion; stale records are not critical.
			_ = db.DeleteChunksByFile(path)
		}
	}

	// Index each discovered file.
	for _, filePath := range files {
		mtime := fileMtime(filePath)
		if !full {
			if rec, ok := existing[filePath]; ok && rec.LastModified >= mtime {
				continue
			}
		}

		n, err := indexFile(db, model, filePath, mtime)
		if err != nil {
			// Log and skip on per-file errors to keep the rest of the run going.
			fmt.Fprintf(os.Stderr, "cc-search: skip %s: %v\n", filePath, err)
			continue
		}
		indexedChunks += n
		indexedFiles++
	}

	return indexedChunks, indexedFiles, nil
}

// findJSONLFiles walks claudeDir/projects and returns paths to all *.jsonl
// files, excluding anything inside a "subagents" directory.
func findJSONLFiles(claudeDir string) []string {
	projectsDir := filepath.Join(claudeDir, "projects")
	if _, err := os.Stat(projectsDir); os.IsNotExist(err) {
		return nil
	}

	var files []string
	filepath.Walk(projectsDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() && info.Name() == "subagents" {
			return filepath.SkipDir
		}
		if !info.IsDir() && strings.HasSuffix(path, ".jsonl") {
			files = append(files, path)
		}
		return nil
	})
	return files
}

// indexFile parses, chunks, embeds, and stores all turns from a single JSONL
// file. It always deletes existing chunks for the file first (re-index
// semantics). Returns the number of chunks inserted.
func indexFile(db *store.DB, model *embedding.Model, filePath string, mtime float64) (int, error) {
	// Always clear stale chunks before re-indexing.
	if err := db.DeleteChunksByFile(filePath); err != nil {
		return 0, fmt.Errorf("delete chunks for %s: %w", filePath, err)
	}

	turns, err := parseJSONLFile(filePath)
	if err != nil {
		// Record the file with its mtime even on parse error so we don't
		// retry on every run. We treat a parse error as "0 chunks indexed".
		_ = db.UpsertFile(filePath, mtime, float64(time.Now().Unix()))
		return 0, fmt.Errorf("parse %s: %w", filePath, err)
	}

	if len(turns) == 0 {
		_ = db.UpsertFile(filePath, mtime, float64(time.Now().Unix()))
		return 0, nil
	}

	project := projectNameFromPath(filePath)

	var allChunks []store.Chunk
	var allTexts []string

	for _, t := range turns {
		textChunks := chunkText(t.Text, model, 400, 100)
		for _, chunk := range textChunks {
			allChunks = append(allChunks, store.Chunk{
				FilePath:  filePath,
				SessionID: t.SessionID,
				Project:   project,
				Role:      t.Role,
				Text:      chunk,
				TurnIndex: t.TurnIndex,
				CreatedAt: t.Timestamp,
			})
			allTexts = append(allTexts, chunk)
		}
	}

	if len(allTexts) == 0 {
		_ = db.UpsertFile(filePath, mtime, float64(time.Now().Unix()))
		return 0, nil
	}

	embeddings, err := model.Embed(allTexts)
	if err != nil {
		return 0, fmt.Errorf("embed %s: %w", filePath, err)
	}

	if err := db.InsertChunks(allChunks, embeddings); err != nil {
		return 0, fmt.Errorf("insert chunks for %s: %w", filePath, err)
	}

	_ = db.UpsertFile(filePath, mtime, float64(time.Now().Unix()))
	return len(allChunks), nil
}

// fileMtime returns the Unix modification time of path as a float64.
// Returns 0 if the file cannot be stat'd.
func fileMtime(path string) float64 {
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return float64(info.ModTime().Unix())
}
