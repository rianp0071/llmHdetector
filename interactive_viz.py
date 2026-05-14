print("Loading heavy machine learning libraries (Torch, Transformers)... This can take 10-30 seconds. Please wait.")

import torch
import numpy as np
from transformer_lens import HookedTransformer
import circuitsvis as cv
import webbrowser
import os

print("Libraries loaded! Now downloading/loading the GPT-2 model...")
model = HookedTransformer.from_pretrained("gpt2-small")

print("\n=== Interactive Transformer Visualizer ===")
print("Type your prompt and press Enter. The visualization will open in your browser.")
print("Type 'quit' to exit.\n")

while True:
    prompt = input("Enter prompt: ")
    if prompt.lower() == 'quit':
        print("Exiting...")
        break
        
    if not prompt.strip():
        continue
        
    layer_str = input("Enter layer to inspect (0-11, default 5): ")
    if not layer_str.strip():
        layer = 5
    else:
        try:
            layer = int(layer_str)
            if layer < 0 or layer > 11:
                print("Layer must be between 0 and 11. Using 5.")
                layer = 5
        except ValueError:
            print("Invalid number. Using 5.")
            layer = 5
            
    print("Running model math...")
    logits, cache = model.run_with_cache(prompt)
    
    # Get the logits for the last token
    next_token_logits = logits[0, -1, :] # [vocab]
    
    # Show top 5 possibilities
    top_5 = torch.topk(next_token_logits, 5)
    print("\nTop 5 possible next words:")
    for i in range(5):
        prob = torch.softmax(next_token_logits, dim=-1)[top_5.indices[i]].item()
        print(f"{i+1}. '{model.to_string(top_5.indices[i])}' ({prob:.2%})")
        
    # NEW: Logit Lens (Layer-by-Layer Prediction)
    print("\n--- Layer-by-Layer Prediction (Logit Lens) ---")
    for l in range(model.cfg.n_layers):
        # Take the residual stream at the end of layer 'l'
        layer_output = cache["resid_post", l][0, -1, :] 
        # Use the model's 'unembedding' to turn that vector into words
        layer_logits = model.unembed(model.ln_final(layer_output))
        top_word = model.to_string(layer_logits.argmax())
        print(f"Layer {l:2}: '{top_word}'")
        
    tokens = model.to_str_tokens(prompt)
    
    # Remove the batch dimension [0] to send [12, seq_len, seq_len]
    attention_pattern = cache["pattern", layer][0] 
    
    print(f"\nGenerating visualization for all 12 heads in Layer {layer}...")
    viz = cv.attention.attention_heads(tokens=tokens, attention=attention_pattern)
    
    html_file = "attention_map.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(str(viz))
        
    print(f"Visualization saved to {os.path.abspath(html_file)}")
    webbrowser.open('file://' + os.path.abspath(html_file))
    print("-" * 40)
