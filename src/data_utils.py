import re
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- Очистка текста ----------
# Удаляем URL и @упоминания
_URL_MENTION_RE = re.compile(r'(?i)\b(?:https?://|www\.)\S+|\B@\w+')
# Оставляем только латиницу, цифры и пробел
_NON_LATIN_RE   = re.compile(r'[^a-z0-9\s]')

def clean_string(s: str) -> str:
    """Простая очистка текста: URL/@, не-латиница, схлопывание пробелов."""
    s = ('' if s is None else str(s)).lower()
    s = _URL_MENTION_RE.sub(' ', s)      # remove URLs and @mentions
    s = _NON_LATIN_RE.sub(' ', s)        # keep only latin letters, digits and spaces
    return re.sub(r'\s+', ' ', s).strip()  # collapse spaces


# ---------- Вспомогательные утилиты ----------
def _human_bytes(n: int) -> str:
    """Красивый вывод размера в байтах."""
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

def _print_done(title: str, start_t: float, extra: str = ""):
    """Единый формат завершения шага с временем выполнения."""
    elapsed = time.perf_counter() - start_t
    print(f"[OK] {title} in {elapsed:.2f}s{(' — ' + extra) if extra else ''}", flush=True)


# ---------- Основная обработка ----------
def process_dataset(
    input_path: str,
    *,
    names = ("tweet_id", "created_at", "query", "user", "text"),
    sep=",",
    encoding="utf-8",
    encoding_errors="replace",
    on_bad_lines="skip",
    test_size=0.10,
    val_size=0.10,
    random_state=42,
    verbose: bool = True,
):
    """
    Читает исходный CSV, чистит колонку `text`, сохраняет:
      - dataset_processed.csv (полностью очищённый датасет)
      - train.csv, val.csv, test.csv (сплиты)
    Файлы пишутся в ту же директорию, что и input_path.
    """
    t0 = time.perf_counter()
    input_path = Path(input_path)
    out_dir = input_path.parent

    # 1) Чтение CSV
    if verbose:
        print("[1/6] Reading source CSV...", flush=True)
    t = time.perf_counter()
    df = pd.read_csv(
        input_path,
        sep=sep,
        header=None,
        names=list(names),
        encoding=encoding,
        encoding_errors=encoding_errors,
        on_bad_lines=on_bad_lines,
        engine="python",  # устойчивее к «грязным» CSV
    )
    mem = int(df.memory_usage(deep=True).sum())
    _print_done("Loaded DataFrame", t, extra=f"shape={df.shape}, memory≈{_human_bytes(mem)}")

    # 2) Очистка текста
    if "text" not in df.columns:
        raise ValueError("В датасете отсутствует колонка 'text'. Проверь имена столбцов.")
    if verbose:
        print("[2/6] Cleaning text column (this can take a while)...", flush=True)
    t = time.perf_counter()

    tqdm.pandas()
    df["text"] = df["text"].progress_apply(clean_string)

    _print_done("Cleaned text", t, extra="with tqdm")

    # 3) Сохранение полного очищённого датасета
    if verbose:
        print("[3/6] Writing dataset_processed.csv...", flush=True)
    t = time.perf_counter()
    processed_path = out_dir / "dataset_processed.csv"
    df.to_csv(processed_path, index=False, encoding="utf-8")
    try:
        size_bytes = processed_path.stat().st_size
        extra = f"size={_human_bytes(size_bytes)}"
    except Exception:
        extra = ""
    _print_done("Saved processed dataset", t, extra=extra)

    # 4) Подготовка индексов для сплитов
    if verbose:
        print("[4/6] Splitting into train/val/test...", flush=True)
    t = time.perf_counter()
    n = len(df)
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    n_val  = int(n * val_size)
    n_train = n - n_test - n_val
    idx_train = idx[:n_train]
    idx_val   = idx[n_train:n_train + n_val]
    idx_test  = idx[n_train + n_val:]
    _print_done("Prepared splits", t, extra=f"train={n_train}, val={n_val}, test={n_test}")

    # 5) Сохранение сплитов
    if verbose:
        print("[5/6] Writing train.csv, val.csv, test.csv...", flush=True)
    t = time.perf_counter()
    train_path = out_dir / "train.csv"
    val_path   = out_dir / "val.csv"
    test_path  = out_dir / "test.csv"
    df.iloc[idx_train].reset_index(drop=True).to_csv(train_path, index=False, encoding="utf-8")
    df.iloc[idx_val].reset_index(drop=True).to_csv(val_path, index=False, encoding="utf-8")
    df.iloc[idx_test].reset_index(drop=True).to_csv(test_path, index=False, encoding="utf-8")
    _print_done("Saved splits", t)

    # 6) Сводка
    if verbose:
        print("[6/6] Done. Summary:", flush=True)
        summary = {
            "processed": str(processed_path),
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "counts": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    return {
        "processed": str(processed_path),
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "counts": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
    }


# ---------- Функции для чтения подготовленных датасетов ----------
def _resolve_dir(path_like: str | Path) -> Path:
    """
    Вспомогательно: принимает путь к исходному CSV ИЛИ к директории ИЛИ к одному из целевых файлов
    и возвращает директорию, в которой ожидаются dataset_processed.csv / train.csv / val.csv / test.csv.
    """
    p = Path(path_like)
    if p.is_dir():
        return p
    # если это один из целевых файлов — вернуть его родителя
    if p.name in {"dataset_processed.csv", "train.csv", "val.csv", "test.csv"}:
        return p.parent
    # иначе считаем, что это исходный CSV — вернём директорию, где лежит исходник
    return p.parent

def read_processed_dataset(path_like: str | Path, *, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Читает dataset_processed.csv из директории (или из директории исходника).
    """
    dir_ = _resolve_dir(path_like)
    f = dir_ / "dataset_processed.csv"
    if not f.exists():
        raise FileNotFoundError(f"Не найден {f}. Сначала запусти process_dataset(...).")
    return pd.read_csv(f, encoding=encoding)

def read_splits(path_like: str | Path, *, encoding: str = "utf-8") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Читает train.csv, val.csv, test.csv из директории (или из директории исходника).
    Возвращает кортеж (df_train, df_val, df_test).
    """
    dir_ = _resolve_dir(path_like)
    train_f = dir_ / "train.csv"
    val_f   = dir_ / "val.csv"
    test_f  = dir_ / "test.csv"
    for f in (train_f, val_f, test_f):
        if not f.exists():
            raise FileNotFoundError(f"Не найден {f}. Сначала запусти process_dataset(...).")
    return (
        pd.read_csv(train_f, encoding=encoding),
        pd.read_csv(val_f, encoding=encoding),
        pd.read_csv(test_f, encoding=encoding),
    )

def truncate(data, ratio):
    size = int(len(data) * ratio)
    return data[:size]


# ---------- Точка входа ----------
if __name__ == "__main__":
    print("Starting dataset processing...", flush=True)
    paths = process_dataset(
        "data/raw_dataset.csv",
        names=("tweet_id", "created_at", "query", "user", "text"),
        sep=",",
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="skip",
        test_size=0.10,
        val_size=0.10,
        random_state=42,
        verbose=True,
    )
    print(
        "Done:\n"
        f"  processed: {paths['processed']}\n"
        f"  train:     {paths['train']} ({paths['counts']['train']} rows)\n"
        f"  val:       {paths['val']} ({paths['counts']['val']} rows)\n"
        f"  test:      {paths['test']} ({paths['counts']['test']} rows)"
    )