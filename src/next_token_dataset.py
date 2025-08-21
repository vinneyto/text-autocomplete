import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class NextTokenDataset(Dataset):
    """
    Пары (X, Y), где:
      X = токены [i : i+seq_size]
      Y = токены [i+1 : i+1+seq_size]
    По желанию добавляем EOS в конец каждого текста. stride — шаг окна.
    """
    def __init__(self,
                 texts,
                 tokenizer,
                 eos_id: int,
                 seq_size: int = 7,
                 max_length: int = 512,
                 *,
                 stride: int = 1):
        assert stride >= 1, "stride должен быть >= 1"
        self.samples = []
        self.seq_size = seq_size
        self.stride = stride

        for line in tqdm(texts):
            if line is None:
                continue

            # если планируем добавить EOS — оставим под него место
            trunc_to = max_length - 1
            token_ids = tokenizer.encode(
                str(line),
                add_special_tokens=False,
                max_length=trunc_to,
                truncation=True,
            )

            token_ids.append(eos_id)

            # нужно как минимум seq_size+1 токенов, чтобы X и Y были длиной seq_size
            if len(token_ids) < seq_size + 1:
                continue

            # скользящее окно со stride
            for i in range(0, len(token_ids) - seq_size):
                x = token_ids[i: i + seq_size]
                y = token_ids[i + 1: i + 1 + seq_size]
                if len(y) == seq_size:
                    self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)