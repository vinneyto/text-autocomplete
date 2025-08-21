import math
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from eval_lstm import evaluate_next_token


def train_next_token(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    epochs=20,
    max_grad_norm=1.0,
    patience=3,
    min_delta_ppl=0.0,   # требуемое улучшение PPL (например, 0.1)
    scheduler=None,      # опционально: ReduceLROnPlateau по val_loss
    restore_best=True
):
    model.to(device)

    best_state = None
    best_ppl = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # ---- TRAIN ----
        model.train()
        running = 0.0

        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch: {epoch}"):
            x_batch = x_batch.to(device)          # [B,T]
            y_batch = y_batch.to(device)          # [B,T]

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)               # [B,T,V]
            V = logits.size(-1)
            loss = criterion(logits.reshape(-1, V), y_batch.reshape(-1))
            loss.backward()

            if max_grad_norm is not None:
                clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            running += loss.item()

        train_loss = running / max(1, len(train_loader))

        # ---- EVAL ----
        val_loss, val_acc = evaluate_next_token(model, val_loader, device, criterion)
        # Perplexity из val_loss; clamp чтобы не словить overflow exp()
        val_ppl = math.exp(min(20.0, val_loss))

        # Шаг LR-шейдюлера по плато (если используешь)
        if scheduler is not None:
            # для ReduceLROnPlateau логично подавать валид. loss
            scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.3f} | "
              f"Val Token Acc: {val_acc:.2%}")

        # ---- EARLY STOPPING по PPL ----
        if best_ppl - val_ppl > min_delta_ppl:
            best_ppl = val_ppl
            epochs_no_improve = 0
            if restore_best:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping: no PPL improvement for {patience} epoch(s). "
                      f"Best Val PPL: {best_ppl:.3f}")
                if restore_best and best_state is not None:
                    model.load_state_dict(best_state)
                return model

    # Доучили все эпохи — вернуть лучшие веса, если нужно
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return model