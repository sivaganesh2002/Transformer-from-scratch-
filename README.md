# Transformer from Scratch

A clean, from-scratch implementation of a Mistral-style decoder-only transformer trained on Wikipedia, featuring every architectural technique used in modern production LLMs — implemented in ~600 lines of readable PyTorch.



---

## Architecture Overview

```
Token IDs
    │
    ▼
┌─────────────────────┐
│   Token Embedding   │  vocab_size → dim
└─────────────────────┘
    │
    ▼  (repeated × n_layers)
┌────────────────────────────────────────────────┐
│  TransformerBlock                              │
│  ┌──────────────┐   ┌───────────────────────┐ │
│  │   RMSNorm    │   │       RMSNorm         │ │
│  └──────┬───────┘   └──────────┬────────────┘ │
│         │                      │              │
│  ┌──────▼───────┐   ┌──────────▼────────────┐ │
│  │  Attention   │   │     SwiGLU FFN        │ │
│  │  GQA + SWA   │   │  (gated activation)   │ │
│  │  + RoPE      │   └──────────┬────────────┘ │
│  └──────┬───────┘              │              │
│         │   residual           │   residual   │
│         └──────────────────────┘              │
└────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────┐
│      RMSNorm        │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   LM Head (Linear)  │  dim → vocab_size  [weight-tied]
└─────────────────────┘
    │
    ▼
  Logits / Loss (CE + Z-loss)
```

---

## Features

| Component | Implementation | Replaces |
|---|---|---|
| Normalisation | RMSNorm | LayerNorm |
| Positional encoding | RoPE (rotary) | Learned absolute embeddings |
| Attention | GQA + Sliding Window | Standard MHA |
| Activation | SwiGLU | ReLU / GELU |
| Inference | KV Cache | Re-computing K/V each step |
| Optimizer | Muon (Newton-Schulz) + AdamW | Adam alone |
| LR schedule | Cosine with linear warmup | Flat LR |
| Loss | Cross-entropy + z-loss | CE only |
| Weight sharing | Embedding tied to LM head | Separate matrices |

---

---

## Quick Start

```bash
git clone https://github.com/<you>/transformer-from-scratch
cd transformer-from-scratch
pip install -r requirements.txt

# Train (downloads ~100MB of Wikipedia)
python train.py

# Generate text
python generate.py --prompt "The French Revolution began when" --max_new 100
```

For the full interactive notebook, open `notebooks/transformer_wikipedia.ipynb` in Google Colab (free T4 GPU).

---

## Mathematics

### 1. RMSNorm

Standard LayerNorm subtracts the mean and divides by standard deviation:

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

RMSNorm removes the mean-centering entirely and uses only the root mean square:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

**Why it works:** The re-scaling invariance property of LayerNorm comes almost entirely from the division by the RMS, not from the mean subtraction. Removing it saves ~7% of the normalisation compute while matching empirical performance (Zhang & Sennrich, 2019). Mistral, LLaMA, and Gemma all use RMSNorm.

---

### 2. Rotary Positional Embeddings (RoPE)

**Problem with absolute positional embeddings:** They encode absolute position, making it hard for the model to generalise to longer sequences than seen during training.

**RoPE encodes position via rotation.** For a query vector $q$ at position $m$ and a key vector $k$ at position $n$, we apply:

$$q_m = R_\Theta^m \, q, \quad k_n = R_\Theta^n \, k$$

where $R_\Theta^m$ is a block-diagonal rotation matrix. The attention score becomes:

$$q_m^\top k_n = q^\top R_\Theta^{n-m} k$$

The score depends only on the **relative position** $n - m$, which is exactly what we want.

**Implementation using complex numbers:** Each consecutive pair of dimensions $(x_{2i}, x_{2i+1})$ is treated as a complex number $z = x_{2i} + i \, x_{2i+1}$. Rotation by angle $\theta$ is multiplication by $e^{i\theta}$:

$$z' = z \cdot e^{i \, m \, \theta_i}$$

where

$$\theta_i = \frac{1}{\Theta^{2i/d}}, \quad \Theta = 10000$$

This is the standard frequency spectrum from the original "Attention is All You Need" positional encoding, now used to rotate rather than add.

**Extrapolation:** Because the model learns to use relative differences, RoPE generalises to longer sequences more gracefully than sinusoidal or learned absolute embeddings.

---

### 3. Grouped Query Attention (GQA)

Standard Multi-Head Attention (MHA) maintains separate $W_K$ and $W_V$ matrices for every head:

$$\text{MHA}: \; n_\text{heads} \text{ Q heads}, \; n_\text{heads} \text{ KV heads}$$

**Problem:** The KV cache grows as $O(n_\text{heads} \cdot T \cdot d_\text{head})$ per layer. At long sequence lengths, this dominates GPU memory.

**GQA solution:** Use fewer KV heads, each shared by a *group* of Q heads:

$$\text{GQA}: \; n_\text{heads} \text{ Q heads}, \; n_\text{kv\_heads} \text{ KV heads}, \quad n_\text{kv\_heads} < n_\text{heads}$$

Before computing attention, K and V are broadcast to match Q:

$$k_\text{expanded} = \text{repeat}(k, \; g) \quad \text{where} \; g = n_\text{heads} / n_\text{kv\_heads}$$

**Memory saving:** The KV cache shrinks by a factor of $g$. In this implementation, $n_\text{heads} = 8$ and $n_\text{kv\_heads} = 2$, giving a $4\times$ KV cache reduction with negligible quality loss.

This is used in Mistral 7B ($n_\text{kv\_heads} = 8$ vs $n_\text{heads} = 32$) and LLaMA 3.

---

### 4. Sliding Window Attention (SWA)

Standard causal attention is $O(T^2)$ — each token attends to all previous tokens. At $T = 4096$, that's 16M attention operations per layer.

**Sliding window:** Each query at position $t$ attends only to positions in $[t - W, t]$:

$$\text{Attention}_\text{SWA}(t) = \text{softmax}\!\left(\frac{q_t \cdot K_{[t-W:t]}^\top}{\sqrt{d}}\right) V_{[t-W:t]}$$

**Complexity:** $O(T \cdot W)$ instead of $O(T^2)$.

**Why it works well:** Most information in language is locally structured. Long-range dependencies propagate through many layers — each layer's window of $W$ tokens covers a total context of $W \times n_\text{layers}$ tokens in the original input.

**Implementation:** A combined causal + window mask is built each forward pass:

```python
mask = 0
mask[k_idx > q_idx]             = -inf   # causal: no future tokens
mask[q_idx - k_idx > window_size] = -inf  # SWA: no tokens too far back
```

This is applied to `F.scaled_dot_product_attention`, which uses Flash Attention internally on CUDA.

**Mistral uses SWA** to achieve 8k effective context with a 4k window.

---

### 5. Scaled Dot-Product Attention

The full attention computation for a single head:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_\text{head}}} + M\right) V$$

where $M$ is the combined causal + sliding window mask ($-\infty$ for masked positions, $0$ elsewhere).

The $1/\sqrt{d_\text{head}}$ scaling prevents the dot products from growing large in magnitude, which would push softmax into regions of near-zero gradient.

In code, `F.scaled_dot_product_attention` fuses the matmuls and softmax, implementing Flash Attention when running on CUDA — reducing HBM reads/writes from $O(T^2 d)$ to $O(T^2)$ via tiling.

---

### 6. SwiGLU Feed-Forward Network

Standard FFN:

$$\text{FFN}(x) = \sigma(xW_1) W_2$$

SwiGLU replaces this with a **gated** variant:

$$\text{SwiGLU}(x) = \big(\text{SiLU}(xW_1) \odot xW_3\big) W_2$$

where $\text{SiLU}(z) = z \cdot \sigma(z)$ (Sigmoid Linear Unit, also called Swish).

**The gate $xW_3$ acts as a learned filter** — it can zero out parts of the activation, giving the network more expressive control over information flow.

**Hidden dimension:** To keep total parameter count comparable to a standard $4 \times d$ FFN, SwiGLU uses:

$$d_\text{hidden} = \left\lceil \frac{8}{3} \cdot d / m \right\rceil \cdot m$$

where $m$ is a hardware-alignment multiple (here 32). This is because SwiGLU uses 3 weight matrices instead of 2.

SwiGLU is used in PaLM, LLaMA, Mistral, and Gemma.

---

### 7. KV Cache

During autoregressive decoding (generating one token at a time), a naïve implementation re-computes $K$ and $V$ for all previous tokens at every step.

**KV cache:** Pre-allocate a buffer and fill it incrementally:

```
Step 1: compute K₁, V₁  → store at index 0
Step 2: compute K₂, V₂  → store at index 1; attend to [K₁, K₂]
Step t: compute Kₜ, Vₜ  → store at index t-1; attend to [K₁, ..., Kₜ]
```

Each step computes K/V only for the **single new token**, not all $t$ tokens. This reduces per-step compute from $O(t \cdot d)$ to $O(d)$ and total generation cost from $O(T^2 d)$ to $O(T d)$.

**Prefill:** The prompt is still processed in a single parallel forward pass (fast), and the resulting K/V tensors populate the cache. Decoding then starts from the last prompt token.

---

### 8. Z-Loss (Logit Stability)

From the PaLM paper (Chowdhery et al., 2022). Adds a small penalty on the log-partition function:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{CE} + \alpha \cdot \mathbb{E}\left[\left(\log \sum_j e^{z_j}\right)^2\right]$$

where $\alpha = 10^{-4}$ and $z$ are the pre-softmax logits.

**Why:** If logits drift large in magnitude, the softmax gradient becomes very small (saturation), causing training instability. Z-loss gently penalises large log-sum-exp values, keeping logits in a regime where gradients flow cleanly. This is especially important for large vocabulary sizes (50k+ tokens).

---

### 9. Weight Tying

The input embedding matrix $W_e \in \mathbb{R}^{V \times d}$ is shared with the output LM head $W_\text{head} \in \mathbb{R}^{d \times V}$:

$$\text{logits} = h \cdot W_e^\top$$

**Benefits:**
- Saves $V \times d$ parameters (~50k × 512 = 25.6M parameters, ~60% of the model in this config).
- Enforces coherence: the embedding that encodes a token as *input* is the same as the vector that recognizes it as a likely *output*, which is semantically sensible.
- Empirically improves perplexity (Press & Wolf, 2017).

---

### 10. Residual Connections and Pre-Norm

Each block uses pre-norm residual connections:

$$h \leftarrow h + \text{Attention}(\text{RMSNorm}(h))$$
$$h \leftarrow h + \text{FFN}(\text{RMSNorm}(h))$$

**Pre-norm** (normalise before the sublayer) is more stable than post-norm (normalise after) for deep models, because gradients flow unimpeded through the residual stream.

**Output projection scaling:** The output projections (`wo` and `w2`) are initialised with smaller variance:

$$\sigma = \frac{0.02}{\sqrt{2 \cdot n_\text{layers}}}$$

This prevents the residual stream from growing too large at initialisation. The factor of $2 \cdot n_\text{layers}$ reflects the two residual additions per block (attention and FFN).

---

## Muon Optimizer

### Motivation

Adam maintains per-parameter second-moment estimates $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ and divides the gradient by $\sqrt{v_t + \epsilon}$ to normalise each scalar independently. This works well but treats all directions equally.

**Muon** goes further: it ensures the *matrix* update has unit operator norm by orthogonalizing the gradient matrix in one step. This enforces a more principled update geometry.

---

### Algorithm

For each 2D weight matrix $W$ at step $t$:

**Step 1 — Nesterov momentum:**

$$b_t = \mu \, b_{t-1} + g_t$$
$$\tilde{g}_t = g_t + \mu \, b_t$$

where $\mu = 0.95$ is the momentum coefficient and $g_t = \nabla_W \mathcal{L}$.

**Step 2 — Orthogonalize:**

$$G_\perp = \text{NewtonSchulz}(\tilde{g}_t)$$

This approximates the polar factor $U$ in the polar decomposition $\tilde{g}_t = U S V^\top$ (i.e., the "nearest orthogonal matrix").

**Step 3 — Scale:**

$$\Delta W = G_\perp \cdot \sqrt{\max(m, n)}$$

This rescaling ensures the update has consistent RMS regardless of matrix dimensions.

**Step 4 — Update:**

$$W \leftarrow W - \eta \, \Delta W$$

---

### Newton-Schulz Orthogonalization

Finding the polar factor of $G$ exactly requires SVD, which is $O(mn \min(m,n))$. Instead, we use a matrix polynomial iteration that converges in $\sim 5$ steps.

**Iteration:**

$$X_0 = G / \|G\|_F$$
$$X_{t+1} = a X_t + b (X_t X_t^\top) X_t + c (X_t X_t^\top)^2 X_t$$

with coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$.

**Derivation:** This is a degree-5 polynomial in $X X^\top$ designed using Chebyshev theory to map all singular values from $[0, 1]$ towards $1$ as fast as possible. After 5 steps, the residual $\|X X^\top - I\|$ is below floating-point precision.

**Cost:** Each iteration requires two matrix multiplications ($X X^\top$ is $m \times m$, the rest are $m \times n$). Total cost: $O(5 \cdot mn)$ — far cheaper than SVD.

---

### Why Muon for LLMs?

The argument from Jordan (2024): for each linear layer, the space of weight matrices has a natural Riemannian geometry (the Stiefel manifold for the row space). The optimal gradient step in this geometry is proportional to the polar factor, not the raw gradient. Muon approximates this optimal step.

Empirically, Muon:
- Trains faster to lower loss than AdamW at equal steps.
- Is particularly effective for weight matrices (as opposed to embeddings and norms, which are better left to AdamW).
- Was used in training the [nanoGPT-speedrun](https://github.com/KellerJordan/modular-muon) record.

**Parameter split in this implementation:**
- **Muon:** all 2D weight matrices (`wq`, `wk`, `wv`, `wo`, `w1`, `w2`, `w3`)
- **AdamW:** embeddings, RMSNorm weights (1D), biases

---

### Cosine LR Schedule

$$\eta(t) = \begin{cases}
\eta_\text{max} \cdot \dfrac{t}{T_\text{warmup}} & t < T_\text{warmup} \\[6pt]
\eta_\text{max} \cdot \max\!\left(r, \; \dfrac{1 + \cos\!\left(\pi \cdot \dfrac{t - T_\text{warmup}}{T_\text{total} - T_\text{warmup}}\right)}{2}\right) & t \geq T_\text{warmup}
\end{cases}$$

where $r = 0.1$ is the minimum LR ratio.

- **Warmup:** Gradually increases LR from 0 to $\eta_\text{max}$ over the first 5% of steps. This prevents large updates early when the model weights are random.
- **Cosine decay:** Smoothly reduces LR, preventing oscillation near convergence.
- **Min ratio:** Keeps a non-zero LR at the end of training (0.1× peak) to allow continued fine-grained learning.

---

## Data Pipeline

### Token Packing

Wikipedia articles vary from ~50 tokens (stubs) to hundreds of thousands of tokens (major articles). Padding each article to a fixed length wastes significant compute.

**Pack strategy:**

```
article_1 tokens ++ <EOS> ++ article_2 tokens ++ <EOS> ++ ...
→ split into non-overlapping windows of (seq_len + 1)
→ x = window[:-1],  y = window[1:]   (causal language modelling target)
```

This achieves near-100% GPU utilisation — every token in every batch is real text, not padding.

### Tokenizer

Uses the GPT-2 BPE tokenizer (50,257 tokens). The vocabulary includes common English subwords and covers Wikipedia well. Swapping in a SentencePiece tokenizer (e.g. the LLaMA tokenizer with 32k tokens) would require only updating `vocab_size` in the config.

---

## Model Configuration

| Parameter | Value | Notes |
|---|---|---|
| `dim` | 512 | Residual stream dimension |
| `n_layers` | 8 | Transformer blocks |
| `n_heads` | 8 | Query heads |
| `n_kv_heads` | 2 | Key/value heads (GQA, 4× fewer) |
| `head_dim` | 64 | `dim / n_heads` |
| `vocab_size` | 50,257 | GPT-2 BPE tokenizer |
| `max_seq_len` | 512 | Maximum context length |
| `window_size` | 256 | SWA window |
| `rope_theta` | 10,000 | RoPE base frequency |
| `ffn_multiple_of` | 32 | Alignment for hardware efficiency |
| **Total params** | ~40M | Including tied embeddings |

### Parameter breakdown

| Component | Parameters |
|---|---|
| Token embedding (`embed`) | 50,257 × 512 ≈ 25.7M |
| Attention per layer (`wq + wk + wv + wo`) | 512² × (8 + 2 + 2 + 8) / 8 = 1.05M |
| FFN per layer (`w1 + w2 + w3`) | ≈ 1.77M |
| RMSNorm per layer | 1,024 |
| LM head | **tied** (= embedding) |
| **Total** | **~40M** |

---

## Generation

### Top-p (Nucleus) Sampling

At each decode step, the vocabulary is sorted by probability. We keep the smallest set $V^* \subseteq V$ such that:

$$\sum_{v \in V^*} p(v) \geq p_\text{top}$$

We then sample from the renormalized distribution over $V^*$ only.

This concentrates probability mass on plausible continuations while preserving diversity, avoiding both:
- **Greedy decoding:** always picks the mode — repetitive, degenerate.
- **Pure sampling:** can produce low-probability nonsense tokens.

Temperature $T$ scales the logits before softmax:

$$p(v) \propto \exp(z_v / T)$$

Lower $T$ → sharper distribution (more conservative). Higher $T$ → flatter (more creative).

---

## Training Details

| Setting | Value |
|---|---|
| Hardware | T4 GPU (Google Colab) |
| Dataset | Wikipedia (1%, ~16k articles) |
| Batch size | 16 sequences × 512 tokens = 8,192 tokens/step |
| Epochs | 2 |
| Muon LR | 0.02 |
| AdamW LR | 3×10⁻⁴ |
| Momentum (Muon) | 0.95 |
| AdamW betas | (0.9, 0.95) |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| LR warmup | 5% of total steps |
| LR min ratio | 10% of peak |
| Z-loss coeff | 10⁻⁴ |

---

## References

1. **Attention is All You Need** — Vaswani et al. (2017). Original transformer architecture.
2. **RoFormer: Enhanced Transformer with Rotary Position Embedding** — Su et al. (2021). RoPE.
3. **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** — Ainslie et al. (2023). Grouped Query Attention.
4. **Mistral 7B** — Jiang et al. (2023). SWA + GQA in practice.
5. **GLU Variants Improve Transformer** — Noam Shazeer (2020). SwiGLU.
6. **Root Mean Square Layer Normalization** — Zhang & Sennrich (2019). RMSNorm.
7. **PaLM: Scaling Language Modeling with Pathways** — Chowdhery et al. (2022). Z-loss.
8. **Using the Output Embedding to Improve Language Models** — Press & Wolf (2017). Weight tying.
9. **Muon: An optimizer for hidden layers in neural networks** — Kosson et al. / Jordan (2024).
10. **Modular Duality in Deep Learning** — Jordan (2024). Geometric motivation for Muon.

---

