# Go Autocomplete Service

This example web service loads an ONNX model and exposes a single HTTP endpoint for text autocompletion.

## Usage

1. Place the exported ONNX model at `models/next_token_64.onnx`.
2. Export the BERT tokenizer used in training:
   ```bash
   python scripts/export_tokenizer.py
   ```
   This saves `bert-base-uncased` files (including `tokenizer.json`) under `go-autocomplete/models/`.
3. Run:
   ```bash
   go run .
   ```
4. Query:
   ```bash
   curl 'http://localhost:8080/autocomplete?prompt=hello'
   ```

The response is a JSON object with the predicted completion.

> The repository does not contain the actual model weights. Export them using `scripts/export_to_onnx.py`.
