import math
import torch
import torch.nn as nn
from tqdm import tqdm

from rouge_score import rouge_scorer

@torch.no_grad()
def evaluate_next_token(model, val_loader, device, criterion, pad_idx = 0):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    total_correct = 0

    for x_batch, y_batch in tqdm(val_loader):
        x_batch = x_batch.to(device)         # [B,T]
        y_batch = y_batch.to(device)         # [B,T]
        logits = model(x_batch)              # [B,T,V]

        V = logits.size(-1)
        loss = criterion(logits.reshape(-1, V), y_batch.reshape(-1))
        total_loss += loss.item()

        # token-acc по непаддингам
        with torch.no_grad():
            pred = logits.argmax(dim=-1)     # [B,T]
            mask = (y_batch != pad_idx)
            total_correct += (pred.eq(y_batch) & mask).sum().item()
            total_tokens  += mask.sum().item()

    val_loss = total_loss / max(1, len(val_loader))
    val_acc  = (total_correct / max(1, total_tokens)) if total_tokens > 0 else 0.0
    return val_loss, val_acc



@torch.no_grad()
def evaluate_next_token_with_rouge(
    model,
    loader,
    device,
    criterion,
    tokenizer,
    decode_skip_special=True
):
    """
    Возвращает: (val_loss, token_acc, rouge1_f, rouge2_f, rougel_f)
    ROUGE считается между tokenizer.decode(pred_ids) и tokenizer.decode(ref_ids).
    """

    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    r1_f_total = 0.0
    r2_f_total = 0.0
    rl_f_total = 0.0
    n_examples = 0

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    for x_batch, y_batch in tqdm(loader, desc="Eval"):
        x_batch = x_batch.to(device)   # [B,T]
        y_batch = y_batch.to(device)   # [B,T]

        logits = model(x_batch)        # [B,T,V]
        V = logits.size(-1)

        loss = criterion(logits.reshape(-1, V), y_batch.reshape(-1))
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)  # [B,T]
        total_correct += (preds == y_batch).sum().item()
        total_tokens  += y_batch.numel()

        preds_cpu = preds.detach().cpu()
        refs_cpu  = y_batch.detach().cpu()

        B, _ = preds_cpu.shape
        for i in range(B):
            pred_ids = preds_cpu[i].tolist()
            ref_ids  = refs_cpu[i].tolist()

            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=decode_skip_special)
            ref_text  = tokenizer.decode(ref_ids,  skip_special_tokens=decode_skip_special)

            scores = scorer.score(ref_text, pred_text)
            r1_f_total += scores["rouge1"].fmeasure
            r2_f_total += scores["rouge2"].fmeasure
            rl_f_total += scores["rougeL"].fmeasure
            n_examples += 1

    avg_loss = total_loss / max(1, len(loader))
    token_acc = total_correct / max(1, total_tokens)

    if n_examples == 0:
        return avg_loss, token_acc, 0.0, 0.0, 0.0

    rouge1_f = r1_f_total / n_examples
    rouge2_f = r2_f_total / n_examples
    rougel_f = rl_f_total / n_examples

    return avg_loss, token_acc, rouge1_f, rouge2_f, rougel_f