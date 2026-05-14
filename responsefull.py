import torch
import numpy as np
from transformer_lens import HookedTransformer
import circuitsvis as cv
import webbrowser
import os

# Load the model
print("Loading model (GPT-2 Small)...")
model = HookedTransformer.from_pretrained("gpt2-small")

prompt = "The capital of France is"
# Fix: HookedTransformer uses 'to_tokens' instead of 'to_ids'
input_ids = model.to_tokens(prompt)

print(f"Starting Generation for: {prompt}")

# We will generate 10 words
for i in range(10):
    # 1. Run the model math on the CURRENT string
    logits = model(input_ids) 
    
    # 2. Get the prediction for the VERY LAST word
    next_token = logits[0, -1, :].argmax().item()
    
    # 3. Append that word to our input (The "Loop")
    input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=-1)
    
    # 4. Print it so we can see the "growth"
    print(f"Word {i+1}: {model.to_string(next_token)}")

print(f"\nFinal Response: {model.to_string(input_ids[0])}")

# After your loop finishes, let's look at the final attention
logits, cache = model.run_with_cache(input_ids)
final_tokens = model.to_str_tokens(input_ids[0])

# Let's look at Layer 10 (the "context" layer)
# We want to see what the VERY LAST word was attending to
last_word_attention = cache["pattern", 10][0, :, -1, :] # [n_heads, query_pos, key_pos]

print("\n--- Context Memory Check ---")
for head in range(12):
    # Find the token (from the prompt) with the highest attention from the final word
    # Indices 1-5 are usually "The capital of France is"
    prompt_attention = last_word_attention[head, 1:6].sum().item()
    print(f"Head {head:2}: Attention to Prompt = {prompt_attention:.2%}")