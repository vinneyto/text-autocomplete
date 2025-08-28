package main

import (
	"encoding/json"
	"io/ioutil"
	"strings"
)

// Tokenizer provides a minimal word-level tokenizer backed by a vocab map.
type Tokenizer struct {
	Vocab    map[string]int
	InvVocab map[int]string
	UnkID    int
}

// LoadTokenizer reads a tokenizer JSON file exported from Python.
func LoadTokenizer(path string) (*Tokenizer, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	t := &Tokenizer{Vocab: raw.Model.Vocab}
	t.InvVocab = make(map[int]string, len(t.Vocab))
	for k, v := range t.Vocab {
		t.InvVocab[v] = k
	}
	if id, ok := t.Vocab["[UNK]"]; ok {
		t.UnkID = id
	}
	return t, nil
}

// Encode converts text into token IDs.
func (t *Tokenizer) Encode(text string) []int64 {
	words := strings.Fields(strings.ToLower(text))
	ids := make([]int64, len(words))
	for i, w := range words {
		if id, ok := t.Vocab[w]; ok {
			ids[i] = int64(id)
		} else {
			ids[i] = int64(t.UnkID)
		}
	}
	return ids
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(ids []int64) string {
	words := make([]string, len(ids))
	for i, id := range ids {
		if w, ok := t.InvVocab[int(id)]; ok {
			words[i] = w
		}
	}
	return strings.Join(words, " ")
}
