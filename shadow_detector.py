"""
Shadow Detector Engine — The core analysis library.
This module is imported by app.py (FastAPI) and can also be run standalone for testing.
All functions return structured data instead of just printing.
"""
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import numpy as np
import sys
import io

# Force UTF-8 output to prevent Windows console encoding crashes with model special tokens
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------------------------------------------------------------------------
# 1. MODEL LOADING — Only runs once when this module is first imported
# ---------------------------------------------------------------------------
print("[Shadow Engine] Loading Qwen-0.5B-Chat... (one-time download)")
model = HookedTransformer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")
print("[Shadow Engine] Model loaded successfully.")


# ---------------------------------------------------------------------------
# 2. CORE: Analyze an existing response (single forward pass)
# ---------------------------------------------------------------------------
def analyze_existing_response(prompt: str, full_response: str) -> list[dict]:
    """
    The core Shadow AI function.
    Takes a prompt + an external AI's response, runs a SINGLE forward pass,
    and returns per-token analysis with entropy, influence, and classification.
    """
    combined_text = prompt + " " + full_response
    tokens = model.to_tokens(combined_text)
    str_tokens = model.to_str_tokens(combined_text)
    prompt_length = len(model.to_str_tokens(prompt))

    # Single forward pass — this is what makes it fast
    logits, cache = model.run_with_cache(tokens)

    # Dynamic layer range based on model architecture
    mid_start = model.cfg.n_layers // 3
    mid_end = (2 * model.cfg.n_layers) // 3
    layer_range = range(mid_start, mid_end)

    results = []

    for i in range(prompt_length, len(str_tokens)):
        token_str = str_tokens[i]

        # A. ENTROPY (Confusion) — higher = model is guessing
        prev_logit = logits[0, i - 1, :]
        probs = F.softmax(prev_logit, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        # B. PROMPT INFLUENCE — how much this token attended to the prompt
        total_prompt_attn = 0
        for l in layer_range:
            prompt_block_attn = cache["pattern", l][0, :, i, 0:prompt_length].sum(dim=-1).mean().item()
            total_prompt_attn += prompt_block_attn
        avg_influence = total_prompt_attn / len(layer_range)

        # C. CLASSIFY
        status = "TRUSTED"
        if avg_influence < 0.05:
            status = "DRIFTING"
        if entropy > 4.5:
            status = "GUESSING"
        if avg_influence < 0.05 and entropy > 5.0:
            status = "HALLUCINATION"

        results.append({
            "token": token_str,
            "influence": round(avg_influence, 4),
            "entropy": round(entropy, 2),
            "status": status,
        })

    return results


# ---------------------------------------------------------------------------
# 3. CONFLICT SCORE — compares what Gemini said vs. what the Shadow expected
# ---------------------------------------------------------------------------
def analyze_conflict(prompt: str, external_response: str) -> list[dict]:
    """
    For each token in the external response, calculate:
    - conflict_score: 1 - P(actual_token) — how surprised the shadow was
    - shadow_preferred: what the shadow model would have said instead
    - top_3: the shadow's top 3 candidates with probabilities
    """
    combined_text = prompt + " " + external_response
    tokens = model.to_tokens(combined_text)
    str_tokens = model.to_str_tokens(combined_text)
    prompt_len = len(model.to_str_tokens(prompt))

    logits, _ = model.run_with_cache(tokens)

    results = []

    for i in range(prompt_len, len(str_tokens)):
        actual_token_id = tokens[0, i]
        actual_token_str = str_tokens[i]

        shadow_probs = F.softmax(logits[0, i - 1, :], dim=-1)

        # Conflict: how unlikely was the actual token?
        prob_of_actual = shadow_probs[actual_token_id].item()
        conflict_score = 1.0 - prob_of_actual

        # Shadow's preferred word
        top_shadow_id = shadow_probs.argmax().item()
        top_shadow_str = model.to_string(top_shadow_id)

        # Top 3 alternatives (for Logit Divergence)
        top_values, top_indices = torch.topk(shadow_probs, 3)
        top_3 = []
        for val, idx in zip(top_values, top_indices):
            top_3.append({
                "token": model.to_string(idx),
                "probability": round(val.item(), 4),
            })

        # Divergence flag: gap between #1 and #2 < 10%
        divergence_gap = (top_values[0] - top_values[1]).item()
        is_diverging = divergence_gap < 0.10

        # Classification
        status = "OK"
        if conflict_score > 0.95:
            status = "HIGH_CONFLICT"
        elif conflict_score > 0.80:
            status = "WARN"

        results.append({
            "token": actual_token_str,
            "conflict_score": round(conflict_score, 4),
            "status": status,
            "shadow_preferred": top_shadow_str.strip(),
            "is_diverging": is_diverging,
            "top_3": top_3,
        })

    return results


# ---------------------------------------------------------------------------
# 4. FULL ANALYSIS — combines everything into one API-ready response
# ---------------------------------------------------------------------------
def full_analysis(prompt: str, response: str) -> dict:
    """
    Runs both analyses and returns a single combined report.
    This is the main function called by the FastAPI endpoint.
    """
    # Run the two analyses
    attention_report = analyze_existing_response(prompt, response)
    conflict_report = analyze_conflict(prompt, response)

    # Merge per-token data
    merged_tokens = []
    for attn, conf in zip(attention_report, conflict_report):
        merged_tokens.append({
            "token": attn["token"],
            "influence": attn["influence"],
            "entropy": attn["entropy"],
            "attention_status": attn["status"],
            "conflict_score": conf["conflict_score"],
            "conflict_status": conf["status"],
            "shadow_preferred": conf["shadow_preferred"],
            "is_diverging": conf["is_diverging"],
            "top_3": conf["top_3"],
        })

    # Calculate overall hallucination score (0-100)
    if merged_tokens:
        avg_conflict = np.mean([t["conflict_score"] for t in merged_tokens])
        high_conflict_count = sum(1 for t in merged_tokens if t["conflict_status"] == "HIGH_CONFLICT")
        hallucination_pct = high_conflict_count / len(merged_tokens)
    else:
        avg_conflict = 0.0
        hallucination_pct = 0.0

    overall_score = round(min(100, (avg_conflict * 50) + (hallucination_pct * 50)), 1)

    verdict = "TRUSTED"
    if overall_score > 60:
        verdict = "LIKELY HALLUCINATING"
    elif overall_score > 35:
        verdict = "SUSPICIOUS"
    elif overall_score > 15:
        verdict = "MOSTLY TRUSTED"

    return {
        "prompt": prompt,
        "response": response,
        "overall_score": overall_score,
        "verdict": verdict,
        "total_tokens": len(merged_tokens),
        "high_conflict_tokens": high_conflict_count if merged_tokens else 0,
        "tokens": merged_tokens,
    }


# ---------------------------------------------------------------------------
# 5. STANDALONE TEST (only runs when executing this file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    user_prompt = "What is the capital of Japan?"
    gemini_response = "The capital of Japan is Tokyo. It is known for its tall skyscrapers and the Eiffel Tower."

    print("\n" + "=" * 70)
    print("RUNNING FULL ANALYSIS (Standalone Test)")
    print("=" * 70)

    report = full_analysis(user_prompt, gemini_response)

    print(f"\nPrompt: {report['prompt']}")
    print(f"Response: {report['response']}")
    print(f"\nOverall Score: {report['overall_score']}/100")
    print(f"Verdict: {report['verdict']}")
    print(f"High Conflict Tokens: {report['high_conflict_tokens']}/{report['total_tokens']}")
    print("-" * 70)

    for t in report["tokens"]:
        flag = ""
        if t["conflict_status"] == "HIGH_CONFLICT":
            flag = " [!!!]"
        elif t["is_diverging"]:
            flag = " [DIV]"
        print(
            f"  {t['token']:15} | conflict={t['conflict_score']:.2f} "
            f"| entropy={t['entropy']:.2f} | influence={t['influence']:.4f} "
            f"| {t['attention_status']:14} | {t['conflict_status']}{flag}"
        )

    # Save visualization
    tokens_list = [t["token"] for t in report["tokens"]]
    conflicts = [t["conflict_score"] for t in report["tokens"]]
    entropies = [t["entropy"] for t in report["tokens"]]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_xlabel("Generated Tokens")
    ax1.set_ylabel("Conflict Score", color="red")
    ax1.bar(range(len(tokens_list)), conflicts, color="salmon", alpha=0.7, label="Conflict")
    ax1.set_xticks(range(len(tokens_list)))
    ax1.set_xticklabels(tokens_list, rotation=45, ha="right")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Entropy", color="blue")
    ax2.plot(range(len(tokens_list)), entropies, color="blue", marker="x", linestyle="--")
    ax2.tick_params(axis="y", labelcolor="blue")

    plt.title(f"Shadow AI Report: '{user_prompt}' — Score: {report['overall_score']}/100 ({report['verdict']})")
    fig.tight_layout()
    plt.savefig("shadow_analysis.png")
    print("\nChart saved to shadow_analysis.png")
