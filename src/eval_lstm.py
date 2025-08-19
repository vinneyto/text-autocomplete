import torch
from typing import Optional

from lstm_model import NextTokenLSTM
from next_token_dataset import resolve_eos_token_id

def _resolve_pad_left_id(tokenizer, model) -> int:
    pid = getattr(tokenizer, "pad_token_id", None)
    if pid is not None:
        return int(pid)
    # используем пад токен из модели, если он задан
    if hasattr(model, "pad_idx") and model.pad_idx is not None:
        return int(model.pad_idx)
    return 0

@torch.no_grad()
def autocomplete_text(
    model: NextTokenLSTM,
    tokenizer,
    text: str,
    *,
    seq_size: int,                  # контекстное окно, как при обучении
    max_new_tokens: int = 50,
    strategy: str = "greedy",       # "greedy" | "sample"
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: Optional[torch.device] = None,
    add_special_tokens: bool = False
) -> str:
    """
    Возвращает автокомплит для входного текста.

    Пример:
        completion = autocomplete_text(model, tok, "Once upon a time", seq_size=128, max_new_tokens=40)
        print(completion)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Превращаем вход в список id; как правило, для LM-дополнения спец-токены не добавляем
    prefix_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)

    eos_id = resolve_eos_token_id(tokenizer)
    pad_left_id = _resolve_pad_left_id(tokenizer, model)

    new_token_ids = model.generate(
        prefix_ids=prefix_ids,
        eos_id=eos_id,
        seq_size=seq_size,
        max_new_tokens=max_new_tokens,
        device=device,
        strategy=strategy,
        temperature=temperature,
        top_k=top_k,
        pad_left_id=pad_left_id,
    )

    # Декодируем только сгенерированную «добавку»
    completion = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    # Небольшая косметика: часто декодер даёт ведущий пробел
    return completion.lstrip()