package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/spf13/cobra"

	"github.com/mikeler216/cc-search/internal/embedding"
	"github.com/mikeler216/cc-search/internal/ingest"
	"github.com/mikeler216/cc-search/internal/search"
	"github.com/mikeler216/cc-search/internal/store"
)

var (
	defaultClaudeDir = filepath.Join(os.Getenv("HOME"), ".claude")
	defaultDBPath    = filepath.Join(defaultClaudeDir, "search-index", "conversations.db")
)

func main() {
	if err := rootCmd().Execute(); err != nil {
		os.Exit(1)
	}
}

func rootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   "cc-search",
		Short: "Semantic search over Claude Code conversation history",
	}
	root.AddCommand(indexCmd(), queryCmd(), statusCmd(), updateCmd())
	return root
}

// isFreshDB returns true if the DB has no meta entries at all (never been
// initialised). This distinguishes a brand-new database from one that has been
// indexed with a different model or schema version.
//
// This is the CLI-level Option B fix for the empty-DB version check: when
// CheckVersion returns false we check whether the meta table is empty. If so,
// we let query/status proceed rather than printing a confusing "Index out of
// date" warning for a DB that has never been indexed.
func isFreshDB(db *store.DB) bool {
	sv, err1 := db.GetMeta("schema_version")
	mh, err2 := db.GetMeta("model_hash")
	return err1 == nil && err2 == nil && sv == "" && mh == ""
}

// checkVersionOrExit performs the version guard used by query and status.
// It returns true if the caller should proceed, false if it already printed an
// error and the caller should exit(1).
func checkVersionOrExit(db *store.DB, modelHash string) bool {
	match, err := db.CheckVersion(modelHash)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cc-search: version check failed: %v\n", err)
		os.Exit(1)
	}
	if !match && !isFreshDB(db) {
		fmt.Fprintln(os.Stderr, "Index out of date. Run `cc-search index` to rebuild.")
		os.Exit(1)
	}
	return true
}

// ── index ──────────────────────────────────────────────────────────────────

func indexCmd() *cobra.Command {
	var dbPath, claudeDir string
	var full, watch bool

	cmd := &cobra.Command{
		Use:   "index",
		Short: "Build or update the search index",
		RunE: func(cmd *cobra.Command, args []string) error {
			model, err := embedding.New()
			if err != nil {
				return err
			}
			defer model.Close()

			if watch {
				return runWatch(dbPath, model, claudeDir)
			}

			chunks, files, err := ingest.Run(dbPath, model, claudeDir, full)
			if err != nil {
				return err
			}

			db, err := store.Open(dbPath)
			if err != nil {
				return err
			}
			defer db.Close()

			totalChunks, totalFiles, _, _ := db.Stats()
			if chunks > 0 || files > 0 {
				fmt.Printf("Indexed %d chunks from %d files.\n", totalChunks, totalFiles)
			} else {
				fmt.Printf("Index up to date. %d chunks from %d files.\n", totalChunks, totalFiles)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&dbPath, "db-path", defaultDBPath, "Path to the search database")
	cmd.Flags().StringVar(&claudeDir, "claude-dir", defaultClaudeDir, "Path to ~/.claude directory")
	cmd.Flags().BoolVar(&full, "full", false, "Force full reindex")
	cmd.Flags().BoolVar(&watch, "watch", false, "Watch for changes and reindex")
	return cmd
}

func runWatch(dbPath string, model *embedding.Model, claudeDir string) error {
	fmt.Println("Watching for changes... (Ctrl+C to stop)")

	projectsDir := filepath.Join(claudeDir, "projects")
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		// Fall back to polling if fsnotify cannot be created at all.
		fmt.Fprintf(os.Stderr, "fsnotify unavailable, falling back to polling: %v\n", err)
		return runPollLoop(dbPath, model, claudeDir)
	}
	defer watcher.Close()

	if err := watcher.Add(projectsDir); err != nil {
		fmt.Fprintf(os.Stderr, "fsnotify watch failed, falling back to polling: %v\n", err)
		return runPollLoop(dbPath, model, claudeDir)
	}

	// Also watch all existing subdirectories so events in nested dirs fire.
	filepath.Walk(projectsDir, func(path string, info os.FileInfo, err error) error {
		if err == nil && info.IsDir() {
			watcher.Add(path)
		}
		return nil
	})

	// Initial index run on start.
	ingest.Run(dbPath, model, claudeDir, false) //nolint:errcheck

	// Debounce timer: start it in a stopped state.
	debounce := time.NewTimer(0)
	<-debounce.C // drain the initial tick so it doesn't fire immediately

	for {
		select {
		case event, ok := <-watcher.Events:
			if !ok {
				return nil
			}
			if strings.HasSuffix(event.Name, ".jsonl") {
				debounce.Reset(2 * time.Second)
			}
		case <-debounce.C:
			ingest.Run(dbPath, model, claudeDir, false) //nolint:errcheck
		case err, ok := <-watcher.Errors:
			if !ok {
				return nil
			}
			fmt.Fprintf(os.Stderr, "watch error: %v\n", err)
		}
	}
}

func runPollLoop(dbPath string, model *embedding.Model, claudeDir string) error {
	for {
		ingest.Run(dbPath, model, claudeDir, false) //nolint:errcheck
		time.Sleep(30 * time.Second)
	}
}

// ── helpers ────────────────────────────────────────────────────────────────

// cwdToProject converts an absolute filesystem path to the project name used
// by Claude Code: /Users/foo/bar → -Users-foo-bar.
func cwdToProject(cwd string) string {
	return "-" + strings.ReplaceAll(strings.TrimLeft(cwd, "/"), "/", "-")
}

// displayProject converts a raw project name back to a human-readable path.
// It walks the filesystem to disambiguate hyphens (e.g. a directory whose
// name contains a hyphen vs. the separator between path components).
func displayProject(rawName string) string {
	segments := strings.Split(strings.TrimLeft(rawName, "-"), "-")
	resolved := "/"
	i := 0
	for i < len(segments) {
		matched := false
		for j := len(segments); j > i; j-- {
			candidate := strings.Join(segments[i:j], "-")
			testPath := filepath.Join(resolved, candidate)
			if info, err := os.Stat(testPath); (err == nil && info.IsDir()) || j == i+1 {
				resolved = testPath
				i = j
				matched = true
				break
			}
		}
		if !matched {
			break
		}
	}
	home := os.Getenv("HOME")
	if strings.HasPrefix(resolved, home+"/") {
		return resolved[len(home)+1:]
	}
	return strings.TrimLeft(resolved, "/")
}

// ── query ──────────────────────────────────────────────────────────────────

func queryCmd() *cobra.Command {
	var dbPath, claudeDir, project, role string
	var top int
	var searchAll bool

	cmd := &cobra.Command{
		Use:   "query <terms...>",
		Short: "Semantic search over conversations",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			queryText := strings.Join(args, " ")

			if !searchAll && project == "" {
				cwd, _ := os.Getwd()
				project = cwdToProject(cwd)
			}

			model, err := embedding.New()
			if err != nil {
				return err
			}
			defer model.Close()

			db, err := store.Open(dbPath)
			if err != nil {
				return err
			}
			defer db.Close()

			checkVersionOrExit(db, model.ModelHash())

			// Run incremental index to pick up any new files before searching.
			ingest.Run(dbPath, model, claudeDir, false) //nolint:errcheck

			results, err := search.Run(db, model, queryText, top, project, role)
			if err != nil {
				return err
			}

			if len(results) == 0 {
				fmt.Println("No results found.")
				return nil
			}

			for i, r := range results {
				fmt.Printf("── Result %d (score: %.2f) %s\n", i+1, r.Score, strings.Repeat("─", 30))
				fmt.Printf("Project: %s\n", displayProject(r.Project))
				fmt.Printf("Session: %s\n", r.SessionID)
				fmt.Printf("Role:    %s\n", r.Role)
				fmt.Printf("Turn:    %d\n", r.TurnIndex)
				fmt.Println()
				text := r.Text
				if len(text) > 300 {
					text = text[:300] + "..."
				}
				for _, line := range strings.Split(text, "\n") {
					fmt.Printf("  %s\n", line)
				}
				fmt.Println()
				fmt.Printf("  → %s\n", r.ResumeCommand)
				fmt.Println()
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&dbPath, "db-path", defaultDBPath, "Path to the search database")
	cmd.Flags().StringVar(&claudeDir, "claude-dir", defaultClaudeDir, "Path to ~/.claude directory")
	cmd.Flags().IntVar(&top, "top", 5, "Number of results")
	cmd.Flags().BoolVar(&searchAll, "all", false, "Search all projects")
	cmd.Flags().StringVar(&project, "project", "", "Filter by project path")
	cmd.Flags().StringVar(&role, "role", "", "Filter by role (user|assistant)")
	return cmd
}

// ── status ─────────────────────────────────────────────────────────────────

func statusCmd() *cobra.Command {
	var dbPath string

	cmd := &cobra.Command{
		Use:   "status",
		Short: "Show index statistics",
		RunE: func(cmd *cobra.Command, args []string) error {
			db, err := store.Open(dbPath)
			if err != nil {
				return err
			}
			defer db.Close()

			model, err := embedding.New()
			if err != nil {
				return err
			}
			defer model.Close()

			checkVersionOrExit(db, model.ModelHash())

			chunks, files, _, err := db.Stats()
			if err != nil {
				return err
			}

			dbSize := int64(0)
			if info, err := os.Stat(dbPath); err == nil {
				dbSize = info.Size()
			}
			fmt.Printf("Chunks:  %d\n", chunks)
			fmt.Printf("Files:   %d\n", files)
			fmt.Printf("DB size: %.1f KB\n", float64(dbSize)/1024)
			return nil
		},
	}
	cmd.Flags().StringVar(&dbPath, "db-path", defaultDBPath, "Path to the search database")
	return cmd
}

// ── update ─────────────────────────────────────────────────────────────────

func updateCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "update",
		Short: "Update cc-search to the latest version",
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Println("Checking for updates...")

			goos := runtime.GOOS
			goarch := runtime.GOARCH
			binaryName := fmt.Sprintf("cc-search-%s-%s", goos, goarch)
			url := fmt.Sprintf("https://github.com/mikeler216/cc-search/releases/latest/download/%s", binaryName)

			resp, err := http.Get(url) //nolint:noctx
			if err != nil {
				return fmt.Errorf("download failed: %w", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				return fmt.Errorf("download failed: HTTP %d", resp.StatusCode)
			}

			self, err := os.Executable()
			if err != nil {
				return err
			}
			self, err = filepath.EvalSymlinks(self)
			if err != nil {
				return err
			}

			tmpPath := self + ".new"
			f, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0755)
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, resp.Body); err != nil {
				f.Close()
				os.Remove(tmpPath)
				return err
			}
			f.Close()

			if err := os.Rename(tmpPath, self); err != nil {
				os.Remove(tmpPath)
				return fmt.Errorf("replace binary: %w", err)
			}

			fmt.Println("Updated successfully. Reindexing...")
			reindex := exec.Command(self, "index", "--full")
			reindex.Stdout = os.Stdout
			reindex.Stderr = os.Stderr
			return reindex.Run()
		},
	}
}
