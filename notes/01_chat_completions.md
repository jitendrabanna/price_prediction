# Chat Completions API

## What Is It?

The Chat Completions API is the primary interface for interacting with large language models (LLMs) like GPT-4o. You submit a *conversation* — a list of messages — and the model returns the next message in that conversation.

---

## The Messages Format

Every request contains a `messages` array. Each item has two required fields:

| Field | Description |
|-------|-------------|
| `role` | Who is speaking: `system`, `user`, or `assistant` |
| `content` | The text of the message |

### Roles Explained

- **`system`** — Sets the assistant's behaviour, persona, or constraints. Appears once at the start. Think of it as a backstage director.
- **`user`** — Represents the human turn. This is the input your application (or end user) provides.
- **`assistant`** — Represents a previous model reply. You include these to give the model context about prior turns.

```python
messages = [
    {"role": "system",    "content": "You are a concise and friendly assistant."},
    {"role": "user",      "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subfield of AI where models learn from data."},
    {"role": "user",      "content": "Can you give me a real-world example?"},
]
```

---

## The API Call

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_tokens=500,      # cap the response length
    temperature=0.7,     # randomness (0 = deterministic, 2 = very random)
)

print(response.choices[0].message.content)
```

---

## The Response Object

```
ChatCompletion(
  id='chatcmpl-...',
  model='gpt-4o-mini-...',
  choices=[
    Choice(
      finish_reason='stop',              # 'stop' | 'length' | 'content_filter'
      message=ChatCompletionMessage(
        role='assistant',
        content='...'
      )
    )
  ],
  usage=CompletionUsage(
    prompt_tokens=42,
    completion_tokens=18,
    total_tokens=60
  )
)
```

### Key Fields

| Field | Meaning |
|-------|---------|
| `choices[0].message.content` | The generated text |
| `choices[0].finish_reason` | Why generation stopped (`stop` = normal, `length` = hit `max_tokens`) |
| `usage.prompt_tokens` | Tokens in your input |
| `usage.completion_tokens` | Tokens in the reply |
| `usage.total_tokens` | Sum — used for billing |

---

## Important Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | — | Which model to use (e.g., `gpt-4o`, `gpt-4o-mini`) |
| `messages` | — | The conversation history |
| `max_tokens` | model max | Cap on completion tokens |
| `temperature` | 1.0 | Sampling temperature (0–2) |
| `top_p` | 1.0 | Nucleus sampling threshold |
| `stop` | None | String(s) that halt generation |
| `stream` | False | Return tokens as they are generated |
| `n` | 1 | Number of completion choices to return |

---

## Real-World Tips

1. **Always include a system prompt.** Even a simple one ("You are a helpful assistant") improves response quality and predictability.
2. **Keep the system prompt focused.** Vague instructions lead to inconsistent behaviour.
3. **Log `usage` in production.** Token counts directly translate to costs.
4. **Check `finish_reason`.** If it's `length`, the reply was cut off — increase `max_tokens` or shorten the prompt.
5. **Use `temperature=0` for tasks that need consistency** (e.g., structured output, classification). Use higher values for creative tasks.

---

## See Also

- [OpenAI Chat Completions Docs](https://platform.openai.com/docs/api-reference/chat)
- [Example code](../examples/01_chat_completions.py)
