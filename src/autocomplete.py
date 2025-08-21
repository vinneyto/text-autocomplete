from typing import Optional
import torch
from lstm_model import NextTokenLSTM

@torch.no_grad()
def autocomplete_text(
    model: NextTokenLSTM,
    tokenizer,
    text: str,
    *,
    eos_id: int,
    seq_size: int,                  # контекстное окно, как при обучении
    max_new_tokens: int = 50,
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

    new_token_ids = model.generate(
        prefix_ids=prefix_ids,
        eos_id=eos_id,
        seq_size=seq_size,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    # Декодируем только сгенерированную «добавку»
    completion = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    # Небольшая косметика: часто декодер даёт ведущий пробел
    return completion.lstrip()