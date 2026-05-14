import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import numpy as np

# 1. Setup Environment
print("Loading Research-Grade Detector (GPT-2 Small)...")
model = HookedTransformer.from_pretrained("gpt2-small")

def run_hallucination_check(prompt, target_noun="France"):
    # Tokenize
    input_ids = model.to_tokens(prompt)
    prompt_tokens = model.to_str_tokens(prompt)
    
    # Find the index of the noun we want to track (the "Anchor")
    try:
        anchor_idx = prompt_tokens.index(f" {target_noun}")
    except ValueError:
        anchor_idx = 1 # Fallback to first real token
        print(f"Warning: '{target_noun}' not found in prompt. Tracking token index 1.")

    results = []
    
    print(f"\nAnalyzing Response Strategy...")
    
    # Generate 15 tokens
    for i in range(15):
        # Run with cache to get internal math
        logits, cache = model.run_with_cache(input_ids)
        
        # A. MEASURE ENTROPY (Confidence)
        # Higher entropy = Model is guessing (High Hallucination Risk)
        last_logit = logits[0, -1, :]
        probs = F.softmax(last_logit, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        
        # B. MEASURE ANCHOR INFLUENCE (Attention)
        # Average attention across all heads in middle layers (where facts live)
        layer_range = range(5, 9)
        total_attn = 0
        for l in layer_range:
            # [batch, head, query, key] -> focus on last word looking at anchor
            total_attn += cache["pattern", l][0, :, -1, anchor_idx].mean().item()
        avg_influence = total_attn / len(layer_range)
        
        # C. GET NEXT WORD
        next_token_id = last_logit.argmax().item()
        next_token_str = model.to_string(next_token_id)
        
        # D. CLASSIFY STATE
        status = "TRUSTED"
        if avg_influence < 0.02: status = "DRIFTING"
        if entropy > 4.5: status = "GUESSING" # Thresholds tuned for GPT2
        
        results.append({
            "token": next_token_str,
            "influence": avg_influence,
            "entropy": entropy,
            "status": status
        })
        
        # Loop back
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=-1)
        print(f"[{status:8}] Word: {next_token_str:12} | Context Influence: {avg_influence:.4f} | Confusion: {entropy:.2f}")

    return results

# --- EXECUTION ---
user_prompt = "The capital of France is"
noun_to_track = "France"
report = run_hallucination_check(user_prompt, noun_to_track)

# --- VISUALIZATION ---
tokens = [r['token'] for r in report]
influences = [r['influence'] for r in report]
entropies = [r['entropy'] for r in report]

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Generated Tokens')
ax1.set_ylabel('Prompt Influence', color='blue')
ax1.plot(tokens, influences, color='blue', marker='o', label='Influence (Higher is better)')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Model Confusion (Entropy)', color='red')
ax2.plot(tokens, entropies, color='red', marker='x', linestyle='--', label='Confusion (Lower is better)')
ax2.tick_params(axis='y', labelcolor='red')

plt.title(f"Hallucination Detection Map: '{user_prompt}'")
fig.tight_layout()
plt.savefig('hallucination_report.png')
print("\nDeep-Analysis Report saved to hallucination_report.png")