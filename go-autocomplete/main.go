package main

import (
	"log"
	"path/filepath"

	"github.com/labstack/echo/v4"
	onnx "github.com/razzie/onnxruntime-go"
)

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

	handler := NewAutocompleteHandler(tokenizer, session)

	e := echo.New()
	e.GET("/autocomplete", handler.Autocomplete)

	log.Println("server started on :8080")
	log.Fatal(e.Start(":8080"))
}
