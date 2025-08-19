import torch
import torch.nn as nn
import torch.nn.functional as F

class NextTokenLSTM(nn.Module):
    """
    LSTM-модель для предсказания следующего токена.
    forward(input_ids) -> logits [B, T, V]
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 1,
        pad_idx: int = 0,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            if self.pad_idx is not None and self.pad_idx < self.embedding.num_embeddings:
                self.embedding.weight[self.pad_idx].fill_(0)

        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]
        return logits: [B, T, V]
        """
        x = self.embedding(input_ids)     # [B,T,E]
        rnn_out, _ = self.rnn(x)          # [B,T,H]
        rnn_out = self.norm(rnn_out)      # [B,T,H]
        rnn_out = self.dropout(rnn_out)   # [B,T,H]
        logits = self.fc(rnn_out)         # [B,T,V]
        return logits

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: list[int],
        *,
        eos_id: int,
        seq_size: int,
        max_new_tokens: int = 50,
        device: torch.device | str = "cpu",
        strategy: str = "greedy",     # "greedy" | "sample"
        temperature: float = 1.0,     # >0; влияет только при strategy="sample"
        top_k: int | None = None,     # обрезка распределения (sample)
        pad_left_id: int | None = None,
    ) -> list[int]:
        """
        Генерируем токены, пока не встретим eos_id или не превысим max_new_tokens.
        На каждом шаге берём logits для последнего таймстепа: logits[:, -1, :].

        strategy="greedy": берём argmax (температура НЕ влияет).
        strategy="sample": применяем softmax( logits / temperature ), опционально top-k,
                           и сэмплируем один токен.
        """
        self.eval()
        device = torch.device(device)
        seq = list(prefix_ids)

        for _ in range(max_new_tokens):
            window = seq[-seq_size:]
            if len(window) < seq_size:
                pad_id = self.pad_idx if pad_left_id is None else pad_left_id
                window = [pad_id] * (seq_size - len(window)) + window

            x = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)  # [1,T]
            logits = self.forward(x)          # [1,T,V]
            next_logits = logits[:, -1, :]    # [1,V]

            if strategy == "sample":
                # temperature: масштабируем «остроту» распределения
                scaled = next_logits / max(1e-6, temperature)
                if top_k is not None and top_k > 0:
                    k = min(top_k, scaled.size(-1))
                    vals, idxs = torch.topk(scaled, k)               # [1,k], [1,k]
                    probs = F.softmax(vals, dim=-1)                   # [1,k]
                    next_id = idxs[0, torch.multinomial(probs[0], 1)].item()
                else:
                    probs = F.softmax(scaled, dim=-1)                 # [1,V]
                    next_id = torch.multinomial(probs[0], 1).item()
            else:  # "greedy"
                next_id = torch.argmax(next_logits, dim=-1).item()

            seq.append(next_id)
            if next_id == eos_id:
                break

        return seq[len(prefix_ids):]