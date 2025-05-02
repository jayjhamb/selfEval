import os
import json
import time
import csv
from typing import List, Dict, Tuple, Optional
from few_shot_examples import FEW_SHOT_EXAMPLES

import numpy as np
from openai import OpenAI

# OpenAI client definition
# Uses environment variable OPENAI_API_KEY if no key is provided
def initialize_client(api_key: Optional[str] = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        key = input("Enter your OpenAI API key: ").strip()
    return OpenAI(api_key=key)

DEFAULT_REPEATS: int = 5
DEFAULT_K: int = 5
DEFAULT_TEMPERATURE: float = 0.7

## Loads the dataset specified (function for json or jsonl files depending on which is in the directory)
## from prompt for input (how many samples wanted)
def load_dataset(path: str, limit: Optional[int] = None) -> List[Dict[str, object]]:
    records = []
    if path.endswith('.jsonl'):
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                if limit and idx >= limit:
                    break
                records.append(json.loads(line))
    else:
        with open(path, 'r') as f:
            data = json.load(f)
        for idx, entry in enumerate(data.get('Data', [])):
            if limit and idx >= limit:
                break
            question = entry.get('Question', '').strip()
            answer_val = entry.get('Answer', {}).get('Value', '').strip()
            records.append({'statement': f"{question} — {answer_val}", 'label': True})
    if not records:
        raise RuntimeError(f"No records loaded from {path}")
    return records

# Build a prompt string including few-shot examples and the target statement
def build_few_shot_prompt(target: str) -> str:
    lines = []
    for example in FEW_SHOT_EXAMPLES:
        lines.append(f"Statement: {example['statement']}")
        lines.append(f"Is this statement True or False? Answer: {example['label']}\n")
    lines.append(f"Statement: {target}")
    lines.append("Is this statement True or False? Answer:")
    return '\n'.join(lines)

# Query the model to get the most likely token and normalized P(True) for a statement
def query_true_probability(
    client: OpenAI,
    statement: str,
    model: str = 'davinci-002',
    max_retries: int = 3,
    backoff_factor: float = 1.0
) -> Tuple[str, float]:
    prompt = build_few_shot_prompt(statement)
    for attempt in range(max_retries):
        try:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
                logprobs=5
            )
            choice = response.choices[0]
            token = choice.text.strip()
            logits = choice.logprobs.top_logprobs[0]
            logp_true = logits.get(' True', logits.get('True', -1e9))
            logp_false = logits.get(' False', logits.get('False', -1e9))
            p_true = float(np.exp(logp_true) / (np.exp(logp_true) + np.exp(logp_false)))
            return token, p_true
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(backoff_factor * (2 ** attempt))
            else:
                raise

# generate k candidate answers and average P(True)
def self_evaluate(
    client: OpenAI,
    statement: str,
    model: str = 'davinci-002',
    repeats: int = DEFAULT_REPEATS,
    k: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE
) -> Tuple[bool, float]:
    sample_prompt = f"Statement: {statement}\nAnswer this question:"
    sample_resp = client.completions.create(
        model=model,
        prompt=sample_prompt,
        max_tokens=10,
        n=k,
        temperature=temperature
    )
    p_vals = []
    for cand in sample_resp.choices:
        combined_stmt = f"{statement} — {cand.text.strip()}"
        _, p = query_true_probability(client, combined_stmt, model)
        p_vals.append(p)
    avg_p = float(np.mean(p_vals))
    return avg_p > 0.5, avg_p

# Compute per-bin calibration stats and overall Expected Calibration Error (ece)
def compute_ece(
    results: List[Dict[str, object]],
    bins: int = 10
) -> Tuple[List[Tuple[float, float, int, Optional[float], Optional[float]]], float]:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    stats = []
    N = len(results)
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        group = [r for r in results if lo <= r['p_true'] < hi]
        if not group:
            stats.append((lo, hi, 0, None, None))
            continue
        avg_conf = float(np.mean([r['p_true'] for r in group]))
        acc = float(np.mean([r['prediction'] == r['label'] for r in group]))
        ece += (len(group) / N) * abs(avg_conf - acc)
        stats.append((lo, hi, len(group), avg_conf, acc))
    return stats, ece

# load data, run evaluations, write outputs
def main():
    client = initialize_client()
    dataset_path = input("Dataset path (JSON/JSONL): ").strip()
    limit = int(input("Sample size [100]: ") or 100)
    repeats = int(input(f"Repeats [default {DEFAULT_REPEATS}]: ") or DEFAULT_REPEATS)
    k = int(input(f"Candidates K [default {DEFAULT_K}]: ") or DEFAULT_K)
    temp = float(input(f"Temperature [default {DEFAULT_TEMPERATURE}]: ") or DEFAULT_TEMPERATURE)
    model = input("Model [davinci-002]: ").strip() or 'davinci-002'
    out_jsonl = input("Results JSONL [results.jsonl]: ").strip() or 'results.jsonl'
    out_csv = input("Summary CSV [summary.csv]: ").strip() or 'summary.csv'
    bins = int(input("ECE bins [10]: ").strip() or 10)

    data = load_dataset(dataset_path, limit)
    print(f"Loaded {len(data)} records.")

    results = []
    with open(out_jsonl, 'w', buffering=1) as fout:
        for rec in data:
            pred, conf = self_evaluate(client, rec['statement'], model, repeats, k, temp)
            entry = {'statement': rec['statement'], 'label': rec['label'], 'prediction': pred, 'p_true': conf}
            fout.write(json.dumps(entry) + "\n")
            results.append(entry)

    bins_stats, ece_val = compute_ece(results, bins)
    accuracy = float(np.mean([r['prediction'] == r['label'] for r in results]))

    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['bin_low', 'bin_high', 'count', 'avg_confidence', 'accuracy'])
        for row in bins_stats:
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(['overall_accuracy', accuracy])
        writer.writerow(['ece', ece_val])

    print(f"Processed {len(results)} examples. Accuracy: {accuracy:.4f}, ECE: {ece_val:.4f}")
    print(f"Outputs -> JSONL: {out_jsonl}, CSV: {out_csv}")

if __name__ == '__main__':
    main()
