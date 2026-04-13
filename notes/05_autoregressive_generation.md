# Autoregressive Generation

## What Does "Autoregressive" Mean?

**Autoregressive** means that each new output depends on (is conditioned on) all previous outputs. LLMs generate text one token at a time, left to right, where every token is predicted based on the full context so far.

```
Input prompt: "The cat sat on the"

Step 1: P(next | "The cat sat on the")           → " mat"
Step 2: P(next | "The cat sat on the mat")        → "."
Step 3: P(next | "The cat sat on the mat.")       → <end>
```

Each step is a complete forward pass through the model. Generation cannot be parallelised across output positions — it is inherently sequential.

---

## The Generation Loop

At each step, the model produces a **probability distribution over the entire vocabulary** (~100,000+ tokens). It then selects the next token using one of several *sampling strategies*:

```
Vocabulary distribution at step t:
  " mat"    → 34%
  " floor"  → 18%
  " couch"  → 11%
  " rug"    → 9%
  ...       → remaining probability
```

The selected token is appended to the sequence, and the loop repeats.

---

## Sampling Strategies

### Greedy Decoding
Always pick the token with the highest probability.

```
→ Deterministic but can be repetitive and "safe"
→ Equivalent to temperature=0 (approximately)
```

### Temperature Sampling
Scale logits before applying softmax:

- **temperature < 1** → sharpen the distribution (more confident, less diverse)
- **temperature = 1** → unchanged distribution
- **temperature > 1** → flatten the distribution (more random, more creative)

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a poem."}],
    temperature=0.9,  # creative
)
```

### Top-p (Nucleus) Sampling
Only sample from the smallest set of tokens whose cumulative probability exceeds `p`:

- **top_p=0.1** → only the top ~10% of probability mass (conservative)
- **top_p=0.9** → top ~90% of probability mass (diverse)
- **top_p=1.0** → full vocabulary (no filtering)

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Name a random planet."}],
    temperature=1.0,
    top_p=0.9,
)
```

> **Best practice**: Adjust `temperature` OR `top_p`, not both simultaneously.

---

## Stop Sequences

A **stop sequence** is a string that signals the model to halt generation:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "List 3 fruits, separated by newlines."}],
    stop=["\n\n"],   # stop at double newline
    max_tokens=50,
)
```

When the model generates the stop sequence, generation ends and `finish_reason` is set to `"stop"`. The stop sequence itself is **not** included in the output.

---

## Finish Reasons

| `finish_reason` | Meaning |
|-----------------|---------|
| `"stop"` | Natural end of generation or stop sequence encountered |
| `"length"` | Reached `max_tokens` — output may be truncated |
| `"content_filter"` | Output blocked by safety filters |
| `"tool_calls"` | Model requested a function/tool call |

Always check `finish_reason` in production:

```python
choice = response.choices[0]
if choice.finish_reason == "length":
    print("Warning: response was truncated!")
```

---

## Implications of Autoregressive Generation

### 1. Latency Scales with Output Length

More output tokens = more autoregressive steps = higher latency. This is why streaming is essential for long responses (users see text appearing rather than waiting).

### 2. Early Tokens Influence Later Ones

Because each token is conditioned on all previous tokens, an unexpected early token can cascade through the entire output. This is why:
- Prompt format matters enormously.
- Chain-of-thought prompting works — reasoning steps "prime" the model for a better answer.

### 3. Non-Determinism

Even with `temperature=0`, results may vary slightly due to floating-point non-determinism across hardware and software versions. Use `seed` (where supported) for reproducibility:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Pick a number from 1 to 10."}],
    temperature=0,
    seed=42,
)
```

### 4. Prompt Position Matters

Text at the beginning and end of the context tends to have more influence on the output than text in the middle — a phenomenon called the **lost in the middle** effect.

---

## Practical Parameters Summary

| Parameter | Range | Effect |
|-----------|-------|--------|
| `temperature` | 0–2 | Controls randomness. 0 ≈ greedy, >1 = creative |
| `top_p` | 0–1 | Nucleus sampling threshold |
| `max_tokens` | 1–model limit | Maximum completion tokens |
| `stop` | string or list | Halt generation on specific strings |
| `seed` | integer | Attempt at reproducibility (best-effort) |
| `frequency_penalty` | -2 to 2 | Penalise repeated tokens (>0 reduces repetition) |
| `presence_penalty` | -2 to 2 | Penalise any previously used tokens (>0 increases diversity) |

---

## Real-World Tips

1. **Use `temperature=0` for structured/deterministic tasks** (JSON extraction, classification).
2. **Use higher temperature for creative tasks** (stories, brainstorming).
3. **Set `max_tokens` conservatively** and increase if needed — prevents runaway costs.
4. **Use stop sequences for structured output** (e.g., stop at `"\n\n"` or `"###"`).
5. **Streaming is preferred for responses >100 tokens** — the UX is dramatically better.
6. **`frequency_penalty=0.5`** is a useful default to reduce repetition without being too aggressive.

---

## See Also

- [Example code](../examples/05_autoregressive_generation.py)
- [Streaming Responses](06_streaming.md)
- [Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
