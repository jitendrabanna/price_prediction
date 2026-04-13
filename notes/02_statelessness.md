# Statelessness of LLMs

## The Core Concept

> **LLMs are stateless.** Every API call is completely independent. The model has no memory of previous requests.

When you make two separate API calls, the model treats them as if they have never interacted before — even if you just finished a long conversation.

---

## Why Are LLMs Stateless?

LLMs are served as **stateless HTTP endpoints**. There is no session concept at the model level:

- The model weights are fixed after training.
- Each inference call is a self-contained forward pass.
- No conversation state is stored on the server between calls.

This is an intentional design choice for:
- **Scalability** — any server can handle any request.
- **Privacy** — the server doesn't need to persist your data.
- **Reproducibility** — given the same input, you get the same (or similar) output.

---

## The Amnesia Problem

```python
# Call 1
response1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "My name is Alex."}],
)

# Call 2 — completely separate, no memory of Call 1
response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is my name?"}],
)
# → "I don't know your name..."
```

---

## The Fix: Manual History Management

The `messages` list **is** the model's memory. To maintain context, you must include all previous turns in every request:

```python
messages = []

# Turn 1
messages.append({"role": "user", "content": "My name is Alex."})
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
reply = response.choices[0].message.content
messages.append({"role": "assistant", "content": reply})

# Turn 2 — history is included
messages.append({"role": "user", "content": "What is my name?"})
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
# → "Your name is Alex."
```

---

## Visual Analogy

```
Without history:          With history:

Call 1: [User: Hi Alex]   Call 1: [User: Hi Alex]
         ↓                          ↓ (save reply)
         ✗ forgotten      Call 2: [User: Hi Alex] [Asst: Hello!] [User: My name?]
                                    ↑ full context included
```

---

## Practical Implications

### 1. Context Window Grows with Each Turn

Every turn you add makes the prompt longer. Long conversations eventually hit the model's context window limit.

**Strategy**: Summarise older turns and replace them with the summary to save tokens.

### 2. Token Costs Scale with History

You pay for *all* tokens sent, including the entire history on every call.

```
Turn 1: 50 prompt tokens
Turn 2: 50 + 80 = 130 prompt tokens  (history + new turn)
Turn 3: 130 + 60 = 190 prompt tokens
```

### 3. External Memory Patterns

For long-running assistants, maintain state externally:

| Pattern | Description |
|---------|-------------|
| Full history | Keep all messages in a list (simplest, uses most tokens) |
| Sliding window | Keep only the last N turns |
| Summarisation | Summarise old turns; keep recent ones verbatim |
| Vector store | Retrieve relevant past messages via semantic search |

---

## Real-World Tips

1. **Persist history in a database** for multi-session chatbots — don't rely on in-process memory.
2. **Implement a token budget**. When `count_tokens(messages) > THRESHOLD`, summarise older turns.
3. **System prompts are NOT remembered automatically** — include them in every call.
4. **Never assume the model "knows" something** unless it's in the current `messages` list or was part of its training data.

---

## See Also

- [Example code](../examples/02_statelessness.py)
- [Context Window & Tokens](03_context_window_tokens.md)
