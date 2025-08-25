import time
from typing import List, Tuple, Callable, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

from rouge_score import rouge_scorer, scoring


def truncate(seq, n: Optional[int]) -> List:
    return list(seq)[:n] if n is not None else list(seq)


@dataclass
class PairingConfig:
    prompt_len: int = 16
    cont_len: int = 32
    max_text_tokens: int = 256
    min_total_tokens: int = 8


class RougeAutocompleteTester:
    def __init__(self, tokenizer, pairing: PairingConfig = PairingConfig(), use_stemmer: bool = True):
        self.tok = tokenizer
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.pairing = pairing
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)

    def _make_pairs_from_texts(self, texts: List[str]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        P, C, M, MIN = (self.pairing.prompt_len, self.pairing.cont_len,
                        self.pairing.max_text_tokens, self.pairing.min_total_tokens)

        for t in texts:
            enc = self.tok(t, truncation=True, max_length=M, add_special_tokens=False)
            ids = enc["input_ids"]
            if len(ids) < max(MIN, P + 1):
                continue
            prompt_ids = ids[:P]
            ref_ids    = ids[P:P + C]
            if not ref_ids:
                continue
            prompt_text = self.tok.decode(prompt_ids, skip_special_tokens=True)
            ref_text    = self.tok.decode(ref_ids,  skip_special_tokens=True)
            # print(prompt_text)
            # print(ref_text);
            # print("-" * 50)
            pairs.append((prompt_text, ref_text))
        return pairs

    def evaluate(self,
             texts: List[str],
             generator_fn: Callable[[str], str],
             limit_pairs: Optional[int] = None) -> Dict[str, float]:
            pairs = self._make_pairs_from_texts(truncate(texts, limit_pairs))
            if not pairs:
                return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "elapsed_ms": 0.0}

            start = time.perf_counter()

            agg = scoring.BootstrapAggregator()
            for prompt, ref in tqdm(pairs):
                pred = generator_fn(prompt)
                scores = self.scorer.score(ref, pred)
                agg.add_scores(scores)

            res = agg.aggregate()
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000.0

            # rouge-score возвращает float, .item() не обязателен; оставлю безопасное приведение:
            return {
                "rouge1": float(res["rouge1"].mid.fmeasure),
                "rouge2": float(res["rouge2"].mid.fmeasure),
                "rougeL": float(res["rougeL"].mid.fmeasure),
                "elapsed_ms": elapsed_ms,
            }