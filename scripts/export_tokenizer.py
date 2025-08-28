import pathlib
from transformers import AutoTokenizer


def main():
    out_dir = pathlib.Path("go-autocomplete/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
