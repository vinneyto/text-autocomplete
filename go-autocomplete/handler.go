package main

import (
	"net/http"

	"github.com/labstack/echo/v4"
	onnx "github.com/razzie/onnxruntime-go"
)

// AutocompleteHandler wraps model session and tokenizer.
type AutocompleteHandler struct {
	tokenizer *Tokenizer
	session   *onnx.Session
}

// NewAutocompleteHandler constructs the handler.
func NewAutocompleteHandler(t *Tokenizer, s *onnx.Session) *AutocompleteHandler {
	return &AutocompleteHandler{tokenizer: t, session: s}
}

// Autocomplete returns next-token prediction for a given prompt.
func (h *AutocompleteHandler) Autocomplete(c echo.Context) error {
	prompt := c.QueryParam("prompt")
	if prompt == "" {
		return c.String(http.StatusBadRequest, "prompt required")
	}

	inputIDs := h.tokenizer.Encode(prompt)
	if len(inputIDs) < 64 {
		pad := make([]int64, 64-len(inputIDs))
		inputIDs = append(inputIDs, pad...)
	} else if len(inputIDs) > 64 {
		inputIDs = inputIDs[len(inputIDs)-64:]
	}

	tensor, err := onnx.NewTensorFromData(inputIDs)
	if err != nil {
		return c.String(http.StatusInternalServerError, err.Error())
	}
	defer tensor.Destroy()

	outputs, err := h.session.Run(map[string]*onnx.Tensor{"input_ids": tensor})
	if err != nil {
		return c.String(http.StatusInternalServerError, err.Error())
	}

	logits := outputs["logits"].Data().([]float32)
	vocabSize := len(h.tokenizer.Vocab)
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
	completion := h.tokenizer.Decode([]int64{bestID})
	return c.JSON(http.StatusOK, map[string]string{"completion": completion})
}
