import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class NextTokenDataset(Dataset):
    """
    Пары (X, Y) длиной seq_size, сформированные из единого потока токенов.
    Поток строится склейкой всех текстов с разделителем <EOT> (или любым другим).
      X = tokens[i : i + seq_size]
      Y = tokens[i+1 : i + 1 + seq_size]

    Параметры:
      texts            — список строк
      tokenizer        — HF токенайзер (реком.: GPT-2 byte-level BPE)
      seq_size         — длина окна
      stride           — шаг окна (>=1)
      eot_token        — строковый спец‑токен‑разделитель между твитами
      add_eos_per_text — если True, в конец КАЖДОГО текста добавляется eos_id
                         (обычно не нужно при наличии <EOT>)
      max_length       — макс. длина одного текста при кодировании (обрезка)
    """
    def __init__(self,
                 texts,
                 tokenizer,
                 seq_size: int = 64,
                 *,
                 stride: int = 1,
                 eot_token: str = "<EOT>",
                 add_eos_per_text: bool = False,
                 eos_id: int | None = None,
                 max_length: int = 512):
        assert stride >= 1, "stride должен быть >= 1"
        self.seq_size = seq_size
        self.stride = stride
        self.samples = []

        # 1) гарантируем наличие спец‑токена <EOT> в токенайзере
        tokenizer.add_special_tokens({"additional_special_tokens": [eot_token]})
        self.eot_id = tokenizer.convert_tokens_to_ids(eot_token)
        self.tokenizer = tokenizer

        # 2) строим единый поток токенов c разделителями
        token_stream = []
        for line in tqdm(texts, desc="Building token stream"):
            if not line:
                continue
            ids = tokenizer.encode(
                str(line),
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
            if add_eos_per_text and eos_id is not None:
                ids.append(eos_id)

            if not ids:
                continue

            token_stream.extend(ids)
            token_stream.append(self.eot_id)  # разделитель между твитами

        # если поток совсем пуст — оставляем пустой датасет
        if len(token_stream) < seq_size + 1:
            self._length = 0
            return

        # 3) скользящее окно по потоку
        for i in tqdm(range(0, len(token_stream) - seq_size, stride), desc="window"):
            x = token_stream[i: i + seq_size]
            y = token_stream[i + 1: i + 1 + seq_size]
            if len(y) == seq_size:
                self.samples.append((x, y))

        self._length = len(self.samples)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)