package store

import (
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	_ "github.com/mattn/go-sqlite3"
)

func init() {
	vec.Auto()
}

const SchemaVersion = "1"

type DB struct {
	conn *sql.DB
}

type FileRecord struct {
	Path         string
	LastModified float64
	LastIndexed  float64
}

type Chunk struct {
	FilePath  string
	SessionID string
	Project   string
	Role      string
	Text      string
	TurnIndex int
	CreatedAt float64
}

type Result struct {
	Chunk
	Score         float32
	ResumeCommand string
}

func Open(dbPath string) (*DB, error) {
	if err := os.MkdirAll(filepath.Dir(dbPath), 0755); err != nil {
		return nil, err
	}
	conn, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, err
	}
	db := &DB{conn: conn}
	if err := db.createTables(); err != nil {
		conn.Close()
		return nil, err
	}
	return db, nil
}

func (db *DB) Close() error {
	return db.conn.Close()
}

func (db *DB) createTables() error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS meta (
			key   TEXT PRIMARY KEY,
			value TEXT
		)`,
		`CREATE TABLE IF NOT EXISTS files (
			path          TEXT PRIMARY KEY,
			last_modified REAL NOT NULL,
			last_indexed  REAL NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS chunks (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			file_path  TEXT NOT NULL,
			session_id TEXT NOT NULL,
			project    TEXT NOT NULL,
			role       TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
			chunk_text TEXT NOT NULL,
			turn_index INTEGER NOT NULL,
			created_at REAL
		)`,
		`CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path)`,
		`CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project)`,
		`CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0 (
			id        INTEGER PRIMARY KEY,
			embedding FLOAT[384] distance_metric=cosine
		)`,
	}
	for _, s := range stmts {
		if _, err := db.conn.Exec(s); err != nil {
			snippet := s
			if len(snippet) > 40 {
				snippet = snippet[:40]
			}
			return fmt.Errorf("exec %q: %w", snippet, err)
		}
	}
	return nil
}

func (db *DB) SetMeta(key, value string) error {
	_, err := db.conn.Exec(
		"INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", key, value)
	return err
}

func (db *DB) GetMeta(key string) (string, error) {
	var value string
	err := db.conn.QueryRow("SELECT value FROM meta WHERE key = ?", key).Scan(&value)
	if err == sql.ErrNoRows {
		return "", nil
	}
	return value, err
}

func (db *DB) InitMeta(modelHash string) error {
	if err := db.SetMeta("schema_version", SchemaVersion); err != nil {
		return err
	}
	return db.SetMeta("model_hash", modelHash)
}

func (db *DB) CheckVersion(modelHash string) (bool, error) {
	sv, err := db.GetMeta("schema_version")
	if err != nil {
		return false, err
	}
	mh, err := db.GetMeta("model_hash")
	if err != nil {
		return false, err
	}
	return sv == SchemaVersion && mh == modelHash, nil
}

func (db *DB) DropAll() error {
	stmts := []string{
		"DROP TABLE IF EXISTS chunks_vec",
		"DROP TABLE IF EXISTS chunks",
		"DROP TABLE IF EXISTS files",
		"DROP TABLE IF EXISTS meta",
	}
	for _, s := range stmts {
		if _, err := db.conn.Exec(s); err != nil {
			return err
		}
	}
	return db.createTables()
}

func serializeF32(vec []float32) []byte {
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func (db *DB) UpsertFile(path string, lastModified, lastIndexed float64) error {
	_, err := db.conn.Exec(
		"INSERT OR REPLACE INTO files (path, last_modified, last_indexed) VALUES (?, ?, ?)",
		path, lastModified, lastIndexed)
	return err
}

func (db *DB) GetFile(path string) (*FileRecord, error) {
	var rec FileRecord
	err := db.conn.QueryRow(
		"SELECT path, last_modified, last_indexed FROM files WHERE path = ?", path,
	).Scan(&rec.Path, &rec.LastModified, &rec.LastIndexed)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &rec, nil
}

func (db *DB) AllFiles() (map[string]FileRecord, error) {
	rows, err := db.conn.Query("SELECT path, last_modified, last_indexed FROM files")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	files := make(map[string]FileRecord)
	for rows.Next() {
		var rec FileRecord
		if err := rows.Scan(&rec.Path, &rec.LastModified, &rec.LastIndexed); err != nil {
			return nil, err
		}
		files[rec.Path] = rec
	}
	return files, rows.Err()
}

func (db *DB) InsertChunks(chunks []Chunk, embeddings [][]float32) error {
	tx, err := db.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	chunkStmt, err := tx.Prepare(`INSERT INTO chunks
		(file_path, session_id, project, role, chunk_text, turn_index, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer chunkStmt.Close()

	vecStmt, err := tx.Prepare("INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)")
	if err != nil {
		return err
	}
	defer vecStmt.Close()

	for i, c := range chunks {
		res, err := chunkStmt.Exec(c.FilePath, c.SessionID, c.Project, c.Role, c.Text, c.TurnIndex, c.CreatedAt)
		if err != nil {
			return err
		}
		id, err := res.LastInsertId()
		if err != nil {
			return err
		}
		if _, err := vecStmt.Exec(id, serializeF32(embeddings[i])); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func (db *DB) DeleteChunksByFile(filePath string) error {
	tx, err := db.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(
		"DELETE FROM chunks_vec WHERE id IN (SELECT id FROM chunks WHERE file_path = ?)",
		filePath); err != nil {
		return err
	}
	if _, err := tx.Exec("DELETE FROM chunks WHERE file_path = ?", filePath); err != nil {
		return err
	}
	return tx.Commit()
}

func (db *DB) Search(queryEmbedding []float32, topK int, project, role string) ([]Result, error) {
	fetchK := topK
	if project != "" || role != "" {
		fetchK = topK * 5
	}

	rows, err := db.conn.Query(`
		SELECT v.id, v.distance, c.file_path, c.session_id, c.project,
		       c.role, c.chunk_text, c.turn_index, c.created_at
		FROM chunks_vec v
		INNER JOIN chunks c ON c.id = v.id
		WHERE v.embedding MATCH ?
		AND k = ?
		ORDER BY v.distance`,
		serializeF32(queryEmbedding), fetchK)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []Result
	for rows.Next() {
		var (
			id       int64
			distance float32
			r        Result
		)
		if err := rows.Scan(&id, &distance, &r.FilePath, &r.SessionID, &r.Project,
			&r.Role, &r.Text, &r.TurnIndex, &r.CreatedAt); err != nil {
			return nil, err
		}
		if project != "" && !strings.HasPrefix(r.Project, project) {
			continue
		}
		if role != "" && r.Role != role {
			continue
		}
		r.Score = 1.0 - distance
		results = append(results, r)
		if len(results) >= topK {
			break
		}
	}
	return results, rows.Err()
}

func (db *DB) Stats() (chunks, files int, sizeKB int64, err error) {
	if err = db.conn.QueryRow("SELECT COUNT(*) FROM chunks").Scan(&chunks); err != nil {
		return
	}
	if err = db.conn.QueryRow("SELECT COUNT(*) FROM files").Scan(&files); err != nil {
		return
	}
	return
}
