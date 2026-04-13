# Context Window & Tokens

## What Is a Token?

A **token** is the atomic unit of text that LLMs process. Text is split into tokens before being fed to the model and decoded from tokens after generation.

- Roughly **1 token ≈ 4 English characters** or **¾ of a word**.
- Common words like "the", "is", "cat" are usually one token.
- Rare words, numbers, and punctuation can be multiple tokens.
- Whitespace and capitalisation affect tokenisation.

```
"Hello, world!" → ["Hello", ",", " world", "!"]  = 4 tokens
"ChatGPT"       → ["Chat", "G", "PT"]             = 3 tokens (approximately)
```

Use the [OpenAI Tokenizer](https://platform.openai.com/tokenizer) to explore interactively.

---

## What Is the Context Window?

The **context window** is the maximum number of tokens a model can process in a single API call — combining *both* input (prompt) and output (completion).

### Current Limits

| Model | Context Window |
|-------|---------------|
| gpt-4o | 128,000 tokens |
| gpt-4o-mini | 128,000 tokens |
| gpt-3.5-turbo | 16,385 tokens |
| claude-3-opus | 200,000 tokens |

> 128,000 tokens ≈ ~96,000 words ≈ a full novel.

---

## Why Context Window Matters

```
┌──────────────────────────────────────────────┐
│               Context Window (128k)           │
│  ┌────────────────────┐  ┌──────────────────┐ │
│  │   Input (prompt)   │  │   Output (reply) │ │
│  │  system + history  │  │   max_tokens     │ │
│  │  + user message    │  │                  │ │
│  └────────────────────┘  └──────────────────┘ │
└──────────────────────────────────────────────┘
      prompt_tokens      +   completion_tokens
               = total_tokens ≤ context window
```

If `prompt_tokens + max_tokens > context_window`, the API raises:
```
openai.BadRequestError: context_length_exceeded
```

---

## Counting Tokens Programmatically

Use the `tiktoken` library to count tokens before sending a request:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")

text = "Explain quantum computing in simple terms."
token_ids = enc.encode(text)
print(len(token_ids))  # → 7
```

### Counting a Messages List

The API adds overhead tokens per message (for role labels and formatting):

```python
def count_messages_tokens(messages, model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    total = sum(
        3 + sum(len(enc.encode(v)) for v in msg.values())
        for msg in messages
    )
    return total + 3  # reply primer
```

---

## Token Costs

OpenAI charges per **1,000 tokens** (input and output priced separately):

| Model | Input (per 1M) | Output (per 1M) |
|-------|---------------|----------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-3.5-turbo | $0.50 | $1.50 |

*(Prices as of mid-2024 — check [openai.com/pricing](https://openai.com/pricing) for current rates)*

---

## Managing the Context Window

### Problem: Long Conversations Hit the Limit

Each turn adds more tokens. After many exchanges, the cumulative prompt may exceed the context window.

### Strategies

| Strategy | Pros | Cons |
|----------|------|------|
| **Sliding window** | Simple | Loses early context |
| **Summarisation** | Compact | Loses detail |
| **Retrieval-Augmented Generation (RAG)** | Precise | Complex to implement |
| **Increase max context model** | No info loss | Higher cost |

### Sliding Window Example

```python
MAX_TOKENS = 2000

while count_messages_tokens(messages) > MAX_TOKENS:
    # Remove the oldest non-system message pair
    messages.pop(1)  # remove oldest user message
    if len(messages) > 1:
        messages.pop(1)  # remove corresponding assistant reply
```

---

## Real-World Tips

1. **Always log `usage` in production** to track costs and detect runaway prompts.
2. **Set `max_tokens` explicitly** to avoid unexpectedly large completions.
3. **Pre-count tokens before large requests** using `tiktoken` — fail gracefully instead of hitting API errors.
4. **Large system prompts are expensive** — every call includes them, so keep them concise.
5. **Context windows are not free** — a 128k-token prompt costs ~850× more in input tokens than a 150-token prompt.

---

## See Also

- [Example code](../examples/03_context_window_tokens.py)
- [Tiktoken on GitHub](https://github.com/openai/tiktoken)
- [OpenAI Tokenizer (interactive)](https://platform.openai.com/tokenizer)
- [Statelessness](02_statelessness.md)
