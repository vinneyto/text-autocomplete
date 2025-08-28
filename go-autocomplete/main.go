package main

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"net/http"
	"path/filepath"

	"strings"

        onnx "github.com/razzie/onnxruntime-go"
)

// Tokenizer implements a minimal word-level tokenizer using a vocab json map.
type Tokenizer struct {
        Vocab    map[string]int
        InvVocab map[int]string
        UnkID    int
}

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

func (t *Tokenizer) Decode(ids []int64) string {
	words := make([]string, len(ids))
	for i, id := range ids {
		if w, ok := t.InvVocab[int(id)]; ok {
			words[i] = w
		}
	}
	return strings.Join(words, " ")
}

func main() {
	modelPath := filepath.Join("models", "next_token_64.onnx")
	tokenizerPath := filepath.Join("models", "tokenizer.json")

	tokenizer, err := LoadTokenizer(tokenizerPath)
	if err != nil {
		log.Fatalf("load tokenizer: %v", err)
	}

	env, err := onnx.NewEnvironment()
	if err != nil {
		log.Fatalf("create env: %v", err)
	}
	session, err := env.NewSession(modelPath)
	if err != nil {
		log.Fatalf("new session: %v", err)
	}
	defer session.Close()

	http.HandleFunc("/autocomplete", func(w http.ResponseWriter, r *http.Request) {
		prompt := r.URL.Query().Get("prompt")
		if prompt == "" {
			http.Error(w, "prompt required", http.StatusBadRequest)
			return
		}

		inputIDs := tokenizer.Encode(prompt)
		// pad to sequence length expected by model
		if len(inputIDs) < 64 {
			pad := make([]int64, 64-len(inputIDs))
			inputIDs = append(inputIDs, pad...)
		} else if len(inputIDs) > 64 {
			inputIDs = inputIDs[len(inputIDs)-64:]
		}

		tensor, err := onnx.NewTensorFromData(inputIDs)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer tensor.Destroy()

		outputs, err := session.Run(map[string]*onnx.Tensor{"input_ids": tensor})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		logits := outputs["logits"].Data().([]float32)
		// take argmax of last position
		vocabSize := len(tokenizer.Vocab)
		start := (64 - 1) * vocabSize
		bestID := int64(0)
		bestVal := float32(-1e9)
		for i := 0; i < vocabSize; i++ {
			val := logits[start+i]
			if val > bestVal {
				bestVal = val
				bestID = int64(i)
			}
		}
		completion := tokenizer.Decode([]int64{bestID})
		resp := map[string]string{"completion": completion}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	log.Println("server started on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
