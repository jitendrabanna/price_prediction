# Tokenization & Tokenizer Mismatch

## What Is Tokenization?

**Tokenization** is the process of converting raw text into a sequence of integer IDs (tokens) that the model can process. The reverse process — converting IDs back to text — is called **detokenization**.

```
Text  →  Tokenizer  →  [Token IDs]  →  Model
Model →  [Token IDs] →  Tokenizer  →  Text
```

Most modern LLMs use **Byte-Pair Encoding (BPE)**, a sub-word tokenisation algorithm. BPE learns a vocabulary of frequent character sequences from a large corpus, allowing it to represent any text using a fixed vocabulary size.

---

## How BPE Works (Simplified)

1. Start with a character-level vocabulary.
2. Count the most frequent adjacent pair of symbols.
3. Merge that pair into a new symbol.
4. Repeat until the vocabulary reaches the target size (e.g., 100,000 tokens).

The result: common words become single tokens; rare words are split into smaller sub-word pieces.

```
"unhappiness"  →  ["un", "hap", "pi", "ness"]   # 4 tokens
"the"          →  ["the"]                          # 1 token
"ChatGPT"      →  ["Chat", "G", "PT"]             # 3 tokens (example)
```

---

## OpenAI Encodings

OpenAI uses the `tiktoken` library. Different models use different encodings:

| Encoding | Models |
|----------|--------|
| `o200k_base` | gpt-4o, gpt-4o-mini |
| `cl100k_base` | gpt-4, gpt-3.5-turbo, text-embedding-ada-002 |
| `p50k_base` | text-davinci-003, code-davinci-002 |
| `r50k_base` | text-davinci-001 (legacy) |

```python
import tiktoken

# By model name (recommended)
enc = tiktoken.encoding_for_model("gpt-4o-mini")

# By encoding name
enc = tiktoken.get_encoding("cl100k_base")
```

---

## Tokenizer Mismatch

**Tokenizer mismatch** occurs when you count tokens using one tokenizer but the model uses a different one. This leads to incorrect cost estimates and potential context-window miscalculations.

### Same Text, Different Token Counts

| Text | `cl100k_base` | `o200k_base` | `p50k_base` |
|------|--------------|-------------|------------|
| `"The price is $99.99."` | 7 | 7 | 8 |
| `"Привет мир"` (Russian) | 6 | 5 | 9 |
| `"こんにちは世界"` (Japanese) | 14 | 7 | 21 |
| `"def fibonacci(n):"` | 6 | 6 | 7 |

> The differences are larger for non-English text, code, and rare characters.

### Why It Happens

Each encoding was trained on a different corpus and has a different vocabulary. When a vocabulary is larger or better suited to a language, it can represent the same text with fewer tokens.

---

## Special Tokens

Special tokens are reserved IDs used for control purposes. They are **not** produced from text but are injected programmatically.

| Token | Purpose |
|-------|---------|
| `<|endoftext|>` | Marks the end of a document |
| `<|im_start|>` | Marks the start of a message (ChatML format) |
| `<|im_end|>` | Marks the end of a message |

By default, `tiktoken` will raise an error if special tokens appear in text. You must explicitly allow them:

```python
enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
# → [100257]  (single special token ID)
```

---

## Byte-Level Fallback

For characters not in the vocabulary (rare Unicode, emojis, etc.), tiktoken falls back to **byte-level encoding** — each byte becomes its own token. This means a single rare character can produce 2–4 tokens.

```python
enc = tiktoken.get_encoding("cl100k_base")

enc.encode("🦄")  # → [9468, 248, 222]  (3 tokens for one emoji)
```

This has practical implications:
- Non-English text may use 2–5× more tokens than equivalent English text.
- Emojis and special characters are expensive in terms of tokens.
- Costs and context usage for multilingual apps must be estimated carefully.

---

## Encoding/Decoding Round-Trip

Tokenization is **lossless** — you can always decode back to the original text:

```python
enc = tiktoken.get_encoding("cl100k_base")

text = "Hello, AI world! 🚀"
token_ids = enc.encode(text)          # encode
decoded = enc.decode(token_ids)       # decode

assert text == decoded  # always True
```

---

## Practical Guidelines

### Always Match the Tokenizer to the Model

```python
# ✅ Correct
enc = tiktoken.encoding_for_model("gpt-4o-mini")

# ❌ Wrong – using a different model's tokenizer
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
```

### Check Token Counts Before Sending

```python
def fits_in_context(messages, model, max_completion_tokens=500):
    enc = tiktoken.encoding_for_model(model)
    prompt_tokens = sum(
        3 + sum(len(enc.encode(v)) for v in msg.values())
        for msg in messages
    ) + 3
    limit = {"gpt-4o-mini": 128_000, "gpt-4o": 128_000}[model]
    return (prompt_tokens + max_completion_tokens) <= limit
```

### Account for Non-English Text

A rough estimate for non-English languages:
- French/Spanish/German: ~1.1–1.3× more tokens than equivalent English
- Russian/Greek/Arabic: ~2–4× more tokens
- Chinese/Japanese/Korean: ~3–6× more tokens (with `cl100k_base`; fewer with `o200k_base`)

---

## Real-World Tips

1. **Never hard-code a character-to-token ratio.** Use `tiktoken` to count precisely.
2. **Re-count tokens after switching models** — the tokenizer may differ.
3. **For third-party APIs** (Anthropic Claude, Mistral, etc.), they have their own tokenizers. Use their official libraries to count tokens.
4. **Budget for non-English content** — it consumes significantly more tokens.

---

## See Also

- [Example code](../examples/04_tokenization.py)
- [Tiktoken on GitHub](https://github.com/openai/tiktoken)
- [Context Window & Tokens](03_context_window_tokens.md)
- [BPE Paper (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909)
