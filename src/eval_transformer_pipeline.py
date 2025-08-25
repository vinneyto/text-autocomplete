from transformers import pipeline

def build_gpt2_pipeline_fn(model_name="distilgpt2",
                           max_new_tokens=32,
                           temperature=0.8,
                           top_p=0.95,
                           device=0):
    # создаём pipeline
    generator = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        device=device
    )
    tok = generator.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def generator_fn(prompt: str) -> str:
        out = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            num_return_sequences=1,
        )
        full = out[0]["generated_text"]
        # отделяем новую часть после prompt
        return full[len(prompt):].strip() if full.startswith(prompt) else full.strip()

    return generator_fn, tok