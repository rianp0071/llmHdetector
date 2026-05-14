import torch
from transformer_lens import HookedTransformer
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Model
model = HookedTransformer.from_pretrained("gpt2-small")

prompt = "Translate the following sentence into Pirate speak: The capital of France is"
input_ids = model.to_tokens(prompt)
prompt_tokens = model.to_str_tokens(prompt)

# 2. Generate Response
print("Generating response...")
for i in range(10):
    logits = model(input_ids)
    next_token = logits[0, -1, :].argmax().item()
    input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=-1)

# 3. Run the final full sequence through the model to get the Attention Cache
full_logits, cache = model.run_with_cache(input_ids)
all_tokens = model.to_str_tokens(input_ids[0])
gen_start_idx = len(prompt_tokens)

# 4. Calculate "Global Influence"
# We average attention across all layers and heads to see the 'Main Signal'
# Attention shape: [batch, head, query_pos, key_pos]
avg_attention = []

for layer in range(model.cfg.n_layers):
    attn = cache["pattern", layer][0] # [heads, query, key]
    avg_attention.append(attn.mean(dim=0)) # Average across 12 heads

# Average across all 12 layers
total_avg_attn = torch.stack(avg_attention).mean(dim=0)

# 5. Extract Influence of 'France' (Index 4) on every generated word
france_idx = 4 # Index of 'France' in the prompt tokens
influence_scores = []
gen_token_labels = []

print("\n--- Influence of 'France' on Generation ---")
for i in range(gen_start_idx, len(all_tokens)):
    score = total_avg_attn[i, france_idx].item()
    token_text = all_tokens[i]
    influence_scores.append(score)
    gen_token_labels.append(token_text)
    print(f"Word: {token_text:15} | 'France' Influence: {score:.4f}")

# 6. Plotting the Decay
plt.figure(figsize=(10, 5))
plt.plot(gen_token_labels, influence_scores, marker='o', linestyle='-', color='blue')
plt.title(f"Influence of '{prompt_tokens[france_idx]}' over the generated response")
plt.xlabel("Generated Words")
plt.ylabel("Attention Weight (Avg across all Heads/Layers)")
plt.grid(True, alpha=0.3)
plt.savefig('influence_chart.png')
print("Chart saved to influence_chart.png")
