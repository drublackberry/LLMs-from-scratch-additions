# Chapter 2 - Tokenization

## Highlights
- Stride is about the next batch not the offset between target and inputs.
- The offset between target and inputs is the sliding window, which seems fixed at +1 token. 
- Embeddings are often trained as part of the LLM. It is a one-hot encoding operation with a linear layer that is trained. Effectively a LUT.

# Chapter 3 - Attention architecture

## Concept
The token embeddings are the input data {x(1), x(2), ..., x(N)}, upon which the attention weights are learned {alpha(1,1), alpha(1,2), ..., alpha(1,N)}. For each pair of token/embedding there is one weight. 

omega(i, j) = x(i)x(j)^T # attention scores
alpha(i, j) = softmax(omega(i,j)) # atention weights 

We compute now the context vector z = {z(1), ..., z(N)}

z(j) = alpha(i, j)x(j) # dot product


## Implementation of self-attention

1) Use a matrix multiplication to compute the attention weights.

Omega = X@X^T 

with:
- N the number of token embeddings in the sequence
- E the embedding dimension (size of embeddings)

X is then a matrix of NxE
Omega is then a matrix of NxN

1) Use softmax across the row direction (each row to sum 1)

Alpha = softmax(omega, dim=1)

Alpha is a matrix of NxN, being N the number of imput token embeddings

3) Now we compute the context vectors using matrix multiplication

Z = Alpha @ X^T

Z is a matrix of size NxE


## Implementation of self-attention with trainable weights

Self-attention is also called sometimes "scaled dot-product attention"

### Query, Key, Value

q(i) = x(i)W_q  # W_q are learnable weights for all vocabulary
k(i) = x(i)W_k  # W_k are learnable weights for all vocabulary
v(i) = x(i)W_i  # W_v are learnable weights for all vocabulary

Dimensions of q, k, v can be the same as input tokens or different. That gives the shape of W_q, W_k and W_v
d : dimension of token embedding x(i)
d_k, d_q, d_v: dimension of query q(i), key k(i), value v(i) embedings
Typically sizes are d_k = d_v = d_q

W_q, W_k, W_v are of size [N x d_k]

### Attention scores
Compute unnormalized attention scores omega(i,j)

omega(i, j) = q(i) * v(j)

in matrix form we compute Omega of size NxN 

Omega = Q @ K^T   # size [N x d_k]

### Attention weights
Compute the attention scores using sofmtax and normalizing by the square root of the embedding dimension

alpha(i, j) = softmax(omega(i, j) / sqrt(d)) 
Alpha = softmax(Omage / sqrt(d))

### Context vector

z(i) = alpha(i,j) * v(j)

Z = Alpha @ V^T

Z is a matrix of size [N x d_v]

### Python class
We encapsulate the weights under a Linear pytorch layer

```python

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # alpha

        context_vec = attn_weights @ values
        return context_vec

```

## Implementing causal attention
Self-attention works by exposed each word (token) with a sequence to each other token in that sequence. This also means that tokens will be context aware of future tokens, in order to prevent that leakage we masked the attention scores matrix (omega) with a triangular matrix. 
Code-wise it is simpler to implement a triangular mask with infite numbers and let the softmax function take care of it, as otherwise the functions should need to be re-normalized.

We also add a dropout layer at the attention weight matrix to prevent overfitting.


```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
attn_weights = self.dropout(attn_weights)
```

## Implementing multi-head causal attentionn
We spin num_heads heads and we concatenate the output of the context vectors. The context vector then will have length [N x d_v * num_heads]

# Chapter 4 - Building a GPT model

## Example input configuration

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
```

## Class that encapsulates the architecture

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```


## Transformer block
It implements a multiple head attention and then uses a pass through a feedfoward network, layer normalization and shortcut connections.

```python
from previous_chapters import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
```


### FeedForward network
Defining the `FeedForward` network by exploding to 4 times the embedding dimension, gating with GELU, and scaling down to embedding dimension.

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
```
### Shortcut connections
It is a technique to solve for vanishing gradients. It adds outputs to inputs and it requires that size of the input is the same as the size of the output.






