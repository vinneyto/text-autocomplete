def resolve_eos_token_id(tokenizer) -> int:
    """Берём конец последовательности из токенайзера: eos → sep. Бросаем ошибку, если нет."""
    for attr in ("eos_token_id", "sep_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None and tid != -1:
            return int(tid)
    raise ValueError("В токенайзере нет eos/sep токена — добавление EOS невозможно.")