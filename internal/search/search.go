// Package search provides the query layer: it embeds the user's query,
// calls the vector store, and attaches claude resume commands to results.
package search

import (
	"github.com/mikeler216/cc-search/internal/embedding"
	"github.com/mikeler216/cc-search/internal/store"
)

// Run embeds query, performs a vector search, and returns the top-K results
// with ResumeCommand populated for each result.
func Run(db *store.DB, model *embedding.Model, query string, topK int, project, role string) ([]store.Result, error) {
	vecs, err := model.Embed([]string{query})
	if err != nil {
		return nil, err
	}
	results, err := db.Search(vecs[0], topK, project, role)
	if err != nil {
		return nil, err
	}
	for i := range results {
		results[i].ResumeCommand = "claude --resume " + results[i].SessionID
	}
	return results, nil
}
