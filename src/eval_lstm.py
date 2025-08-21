import torch
from tqdm import tqdm

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