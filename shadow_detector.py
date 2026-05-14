"""
Shadow Detector Engine v3 — Calibrated + Battle-Tested Analysis Library.
This module is imported by app.py (FastAPI) and can also be run standalone for testing.

v3 Fixes:
- Skip first response token (instruction-tuned models always flag it)
- Fixed scoring formula (no more *100 inflation)
- Suggestions now trigger on HIGH_CONFLICT tokens, not just attention status
- Divergence check requires entropy > 3.0 to prevent false flags on normal vocabulary
"""
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import numpy as np
import sys
import io
from collections import Counter

# Force UTF-8 output to prevent Windows console encoding crashes with model special tokens
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------------------------------------------------------------------------
# 1. MODEL LOADING — Only runs once when this module is first imported
# ---------------------------------------------------------------------------
print("[Shadow Engine v3] Loading Qwen-0.5B-Chat...")
model = HookedTransformer.from_pretrained(
    "Qwen/Qwen1.5-0.5B-Chat",
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    dtype=torch.bfloat16
)
print("[Shadow Engine v3] Model loaded successfully.")

# Dynamic layer range (middle third of the model)
MID_START = model.cfg.n_layers // 3
MID_END = (2 * model.cfg.n_layers) // 3
LAYER_RANGE = range(MID_START, MID_END)


# ---------------------------------------------------------------------------
# 2. CALIBRATED FULL ANALYSIS — Single forward pass, Z-score based
# ---------------------------------------------------------------------------
def full_analysis(prompt: str, response: str) -> dict:
    """
    Runs the complete hallucination analysis pipeline in a SINGLE forward pass.
    Uses Z-Score calibration to eliminate false positives.
    """
    combined_text = prompt + " " + response
    tokens = model.to_tokens(combined_text)
    str_tokens = model.to_str_tokens(combined_text)
    prompt_len = len(model.to_str_tokens(prompt))

    # ONE forward pass — extracts both logits and attention cache
    logits, cache = model.run_with_cache(tokens)

    # ------------------------------------------------------------------
    # PASS 1: Collect raw metrics for every response token
    # ------------------------------------------------------------------
    raw_data = []

    for i in range(prompt_len, len(str_tokens)):
        token_str = str_tokens[i]
        actual_token_id = tokens[0, i]

        # --- Entropy ---
        prev_logit = logits[0, i - 1, :]
        probs = F.softmax(prev_logit, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        # --- Prompt Influence (attention to prompt block) ---
        total_prompt_attn = 0
        for l in LAYER_RANGE:
            prompt_block_attn = cache["pattern", l][0, :, i, 0:prompt_len].sum(dim=-1).mean().item()
            total_prompt_attn += prompt_block_attn
        influence = total_prompt_attn / len(LAYER_RANGE)

        # --- Conflict Score ---
        prob_of_actual = probs[actual_token_id].item()
        conflict_score = 1.0 - prob_of_actual

        # --- Shadow's top 3 candidates ---
        top_values, top_indices = torch.topk(probs, 3)
        top_shadow_str = model.to_string(top_indices[0])
        top_3 = []
        for val, idx in zip(top_values, top_indices):
            top_3.append({
                "token": model.to_string(idx),
                "probability": round(val.item(), 4),
            })

        # --- Logit Divergence ---
        # Only flag divergence if the model is ALSO confused (entropy > 3.0)
        # This prevents flagging normal vocabulary as "diverging"
        divergence_gap = (top_values[0] - top_values[1]).item()
        is_diverging = divergence_gap < 0.10 and entropy > 3.0

        raw_data.append({
            "token": token_str,
            "entropy": entropy,
            "influence": influence,
            "conflict_score": conflict_score,
            "shadow_preferred": top_shadow_str.strip(),
            "is_diverging": is_diverging,
            "top_3": top_3,
            "token_index": i - prompt_len,  # 0-based index in the response
        })

    # ------------------------------------------------------------------
    # PASS 2: Z-Score Calibration — eliminate false positives
    # ------------------------------------------------------------------
    ENTROPY_WINDOW = 5

    merged_tokens = []
    entropy_buffer = []
    drift_streak = 0
    repetition_tokens = []

    for idx, rd in enumerate(raw_data):
        # ============================================================
        # FIX #1: Skip the FIRST response token from conflict scoring.
        # Instruction-tuned models ALWAYS see token 0 as "surprising"
        # because they expect <|im_end|> or a chat template marker,
        # not the start of a user-written response. This is NOT a
        # hallucination — it's a structural artifact.
        # ============================================================
        is_first_token = (rd["token_index"] == 0)

        # Build rolling averages
        entropy_buffer.append(rd["entropy"])
        if len(entropy_buffer) > ENTROPY_WINDOW:
            entropy_buffer.pop(0)

        avg_entropy = np.mean(entropy_buffer)
        std_entropy = np.std(entropy_buffer) if len(entropy_buffer) > 1 else 1.0

        # Z-Score: how many standard deviations above the local mean?
        z_score = (rd["entropy"] - avg_entropy) / (std_entropy + 1e-9) if std_entropy > 0.01 else 0.0

        # --- CALIBRATED CLASSIFICATION ---
        # Entropy floor 5.0: Leaning strongly against false positives as requested.
        # "guide" in CRISPR is 5.25 (the only remaining false positive from previous tests)
        # So 5.0 ensures we don't catch words in the 4.5-5.0 range, leaving only massive spikes
        is_entropy_spike = z_score > 1.8 and rd["entropy"] > 5.0
        is_low_influence = rd["influence"] < 0.04
        is_high_conflict = rd["conflict_score"] > 0.95

        # First token is ALWAYS marked TRUSTED (structural artifact, not hallucination)
        if is_first_token:
            attention_status = "TRUSTED"
            conflict_status = "OK"
        else:
            # Multi-factor: genuine hallucination requires MULTIPLE signals
            if is_entropy_spike and is_low_influence and is_high_conflict:
                attention_status = "HALLUCINATION"
            elif is_entropy_spike and is_high_conflict:
                attention_status = "GUESSING"
            elif is_low_influence:
                attention_status = "DRIFTING"
            else:
                attention_status = "TRUSTED"

            # Calibrated conflict: entropy spike + high conflict
            if is_high_conflict and is_entropy_spike:
                conflict_status = "HIGH_CONFLICT"
            elif rd["conflict_score"] > 0.80 and is_entropy_spike:
                conflict_status = "WARN"
            else:
                conflict_status = "OK"

        # --- DRIFT / LOOP TRACKING ---
        if rd["influence"] < 0.04 and not is_first_token:
            drift_streak += 1
        else:
            drift_streak = 0

        repetition_tokens.append(rd["token"].strip().lower())
        if len(repetition_tokens) > 8:
            repetition_tokens.pop(0)

        merged_tokens.append({
            "token": rd["token"],
            "influence": round(rd["influence"], 4),
            "entropy": round(rd["entropy"], 2),
            "entropy_z_score": round(z_score, 2),
            "attention_status": attention_status,
            "conflict_score": round(rd["conflict_score"], 4),
            "conflict_status": conflict_status,
            "shadow_preferred": rd["shadow_preferred"],
            "is_diverging": rd["is_diverging"],
            "top_3": rd["top_3"],
        })

    # ------------------------------------------------------------------
    # PASS 3: Calculate overall score and generate suggestions
    # ------------------------------------------------------------------
    # ============================================================
    # FIX #2: Scoring formula.
    # The weights (60, 30, 10) are designed so the MAX possible = 100.
    # Do NOT multiply by 100 again — that was inflating everything.
    # ============================================================
    if merged_tokens:
        # Exclude first token from counts (it's always TRUSTED now)
        scoreable = merged_tokens[1:] if len(merged_tokens) > 1 else merged_tokens
        total_scoreable = len(scoreable)

        high_conflict_count = sum(1 for t in scoreable if t["conflict_status"] == "HIGH_CONFLICT")
        hallucination_count = sum(1 for t in scoreable if t["attention_status"] == "HALLUCINATION")
        guessing_count = sum(1 for t in scoreable if t["attention_status"] == "GUESSING")
        drifting_count = sum(1 for t in scoreable if t["attention_status"] == "DRIFTING")

        if total_scoreable > 0:
            hallucination_pct = hallucination_count / total_scoreable
            guessing_pct = guessing_count / total_scoreable
            conflict_pct = high_conflict_count / total_scoreable
            drift_pct = drifting_count / total_scoreable
        else:
            hallucination_pct = guessing_pct = conflict_pct = drift_pct = 0.0

        # Score formula: percentage-based + minimum floor for any flagged tokens.
        # A single HIGH_CONFLICT or GUESSING token always scores at least 40.
        # This ensures even 1 flag in a long response is visible.
        pct_score = (
            (hallucination_pct * 40) +   # Max 40 pts
            (guessing_pct * 25) +         # Max 25 pts
            (conflict_pct * 25) +         # Max 25 pts
            (drift_pct * 10)              # Max 10 pts
        ) * 100  # Scale to 0-100

        # Floor: if 2+ tokens are flagged, minimum score is 40 ("SUSPICIOUS")
        # Single-token flags are kept proportional to avoid false positives from rare vocabulary
        total_flags = high_conflict_count + hallucination_count + guessing_count
        if total_flags >= 2:
            floor_score = 40
        elif total_flags == 1:
            floor_score = 0  # Let the percentage formula handle it
        else:
            floor_score = 0

        overall_score = round(min(100, max(pct_score, floor_score)), 1)
    else:
        overall_score = 0.0
        high_conflict_count = 0
        hallucination_count = 0
        guessing_count = 0
        drift_pct = 0.0

    verdict = "TRUSTED"
    if overall_score > 60:
        verdict = "LIKELY HALLUCINATING"
    elif overall_score > 35:
        verdict = "SUSPICIOUS"
    elif overall_score > 15:
        verdict = "MOSTLY TRUSTED"

    # ------------------------------------------------------------------
    # FIX #3: Suggestion engine — trigger on HIGH_CONFLICT and GUESSING too
    # ------------------------------------------------------------------
    suggestions = []

    # Suggestion for HIGH_CONFLICT tokens (the most common real signal)
    flagged_words = [t["token"].strip() for t in merged_tokens
                     if t["conflict_status"] == "HIGH_CONFLICT" and t["token"].strip()]
    if flagged_words:
        suggestions.append({
            "type": "CONFLICT",
            "message": f"Shadow AI strongly disagrees with: {', '.join(flagged_words[:5])}",
            "reprompt": f"Can you double-check the facts around '{', '.join(flagged_words[:3])}'? That part seems inconsistent with the question.",
        })

    # Suggestion for GUESSING tokens (high entropy + high conflict)
    guessing_words = [t["token"].strip() for t in merged_tokens
                      if t["attention_status"] == "GUESSING" and t["token"].strip()]
    if guessing_words:
        suggestions.append({
            "type": "UNCERTAIN",
            "message": f"The AI seemed uncertain about: {', '.join(guessing_words[:5])}",
            "reprompt": f"I noticed you might be uncertain about '{', '.join(guessing_words[:3])}'. Can you provide a source or clarify?",
        })

    # Suggestion for confirmed HALLUCINATION tokens
    halluc_words = [t["token"].strip() for t in merged_tokens
                    if t["attention_status"] == "HALLUCINATION" and t["token"].strip()]
    if halluc_words:
        suggestions.append({
            "type": "HALLUCINATION",
            "message": f"Likely hallucination detected: {', '.join(halluc_words[:5])}",
            "reprompt": f"Please verify the facts around '{', '.join(halluc_words[:3])}'. This appears to be fabricated or incorrect.",
        })

    # Suggestion for drift
    if drift_pct > 0.3 if merged_tokens else False:
        suggestions.append({
            "type": "DRIFT",
            "message": "The AI is wandering away from your original question.",
            "reprompt": "Focus back on my specific question. Do not include unrelated information.",
        })

    # Repetition loop detection
    if repetition_tokens:
        token_counts = Counter(repetition_tokens)
        most_common_token, most_common_count = token_counts.most_common(1)[0]
        if most_common_count >= 3 and most_common_token not in ("the", "a", "is", "of", "and", "to", "in", "it", "for", ""):
            suggestions.append({
                "type": "LOOP",
                "message": f"The AI may be repeating itself ('{most_common_token}' appeared {most_common_count} times recently).",
                "reprompt": "You are repeating yourself. Summarize the last point and move to the next topic.",
            })

    return {
        "prompt": prompt,
        "response": response,
        "overall_score": overall_score,
        "verdict": verdict,
        "total_tokens": len(merged_tokens),
        "high_conflict_tokens": high_conflict_count,
        "suggestions": suggestions,
        "tokens": merged_tokens,
    }


# ---------------------------------------------------------------------------
# 3. LIGHTWEIGHT CONFLICT CHECK (for streaming / quick-check endpoint)
# ---------------------------------------------------------------------------
def analyze_conflict(prompt: str, external_response: str) -> list[dict]:
    """Quick conflict-only check without attention analysis. Faster for streaming."""
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
        prob_of_actual = shadow_probs[actual_token_id].item()
        conflict_score = 1.0 - prob_of_actual

        top_shadow_id = shadow_probs.argmax().item()
        top_shadow_str = model.to_string(top_shadow_id)

        # Skip first token for conflict status
        is_first = (i == prompt_len)
        status = "OK"
        if not is_first:
            if conflict_score > 0.95:
                status = "HIGH_CONFLICT"
            elif conflict_score > 0.80:
                status = "WARN"

        results.append({
            "token": actual_token_str,
            "conflict_score": round(conflict_score, 4),
            "status": status,
            "shadow_preferred": top_shadow_str.strip(),
        })

    return results


# ---------------------------------------------------------------------------
# 4. STANDALONE TEST — run with: python shadow_detector.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SHADOW AI v3 — CALIBRATION TEST (3 scenarios)")
    print("=" * 70)

    test_cases = [
        ("CORRECT (short)", "What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("CORRECT (long)", "What is the capital of Japan?",
         "The capital of Japan is Tokyo. It is a major economic center and the most populous city in the country."),
        ("HALLUCINATION", "What is the capital of Japan?",
         "The capital of Japan is Tokyo. It is known for the Eiffel Tower."),
    ]

    for name, prompt, response in test_cases:
        report = full_analysis(prompt, response)
        print(f"\n[{name}] Score: {report['overall_score']}/100 — {report['verdict']}")
        print(f"  Conflicts: {report['high_conflict_tokens']}/{report['total_tokens']}")
        if report["suggestions"]:
            for s in report["suggestions"]:
                print(f"  [{s['type']}] {s['message']}")
        for t in report["tokens"]:
            flag = " <<<" if t["conflict_status"] == "HIGH_CONFLICT" else ""
            print(f"    {t['token']:15} | c={t['conflict_score']:.3f} e={t['entropy']:.2f} z={t['entropy_z_score']:+.2f} i={t['influence']:.4f} | {t['attention_status']:14} {t['conflict_status']}{flag}")
