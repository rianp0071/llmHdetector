import numpy as np

def softmax(x):
    # Standard softmax: exp(x) / sum(exp(x))
    # We subtract max(x) for numerical stability (prevents overflow)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(X, W_Q, W_K, W_V):
    """
    X: Input matrix (Size: Seq_Len x Embedding_Dim)
    W_matrices: The learned weights (Size: Embedding_Dim x Head_Dim)
    """
    # 1. Project inputs into Query, Key, and Value spaces
    Q = np.dot(X, W_Q)
    K = np.dot(X, W_K)
    V = np.dot(X, W_V)
    
    # 2. Calculate raw attention scores (Similarity)
    # Result size: Seq_Len x Seq_Len
    matmul_qk = np.dot(Q, K.T)
    
    # 3. Scale the scores
    d_k = Q.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)
    
    # 4. Softmax to get weights (Rows sum to 1)
    attention_weights = softmax(scaled_attention_logits)
    
    # 5. Multiply weights by Value to get final representation
    output = np.dot(attention_weights, V)
    
    return output, attention_weights

# --- EXAMPLE CASE ---
# Sentence: "The Aggies Won" (3 words)
# Let's assume Embedding Dimension = 4
# X matrix (3 words x 4 dimensions)
X = np.array([
    [1.0, 0.0, 0.2, 0.1], # The
    [0.1, 1.1, 0.0, 0.8], # Aggies
    [0.0, 0.2, 1.5, 0.9]  # Won
])

# Initialize random weights (In a real model, these are learned via Calculus)
np.random.seed(42)
W_Q = np.random.randn(4, 4)
W_K = np.random.randn(4, 4)
W_V = np.random.randn(4, 4)

# Run the attention mechanism
output, weights = scaled_dot_product_attention(X, W_Q, W_K, W_V)

print("Attention Weights (How much each word looks at the others):")
print(weights)
print("\nNew Contextual Vector for 'Aggies':")
print(output[1])