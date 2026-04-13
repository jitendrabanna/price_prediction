# Streaming Responses

## What Is Streaming?

In **non-streaming mode**, the API waits until the model finishes generating the entire response, then sends it all at once. In **streaming mode**, the API sends tokens to the client as soon as they are generated — one chunk at a time.

```
Non-streaming:
  Client ——— request ——→ Server
  Client ←—— (waits) ——— ...generating...
  Client ←— full reply — Server  (3–30s later)

Streaming (Server-Sent Events):
  Client ——— request ——→ Server
  Client ←— chunk 1  ——— (first token: ~200ms)
  Client ←— chunk 2  ———
  Client ←— chunk 3  ———
  ...
  Client ←— [DONE]   ———
```

---

## Why Use Streaming?

| Metric | Non-Streaming | Streaming |
|--------|--------------|-----------|
| Time to first token (TTFT) | Full generation time (e.g., 5s) | ~200–500ms |
| Perceived responsiveness | Low | High |
| Implementation complexity | Simple | Slightly more complex |
| Suitable for long responses | Poor UX | Excellent UX |

Streaming is the standard in modern AI chat interfaces (ChatGPT, Claude, Copilot) precisely because it feels instantaneous.

---

## How to Enable Streaming

Pass `stream=True` to the API call:

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

The return value is an **iterator** of `ChatCompletionChunk` objects rather than a single `ChatCompletion`.

---

## The Chunk Object

Each chunk has this structure:

```
ChatCompletionChunk(
  id='chatcmpl-...',
  model='gpt-4o-mini-...',
  choices=[
    Choice(
      delta=ChoiceDelta(
        role='assistant',   # only set in the first chunk
        content=' Hello'    # incremental text (None when empty)
      ),
      finish_reason=None    # None until the last chunk
    )
  ]
)
```

The **last chunk** has `finish_reason` set (e.g., `"stop"` or `"length"`):

```python
for chunk in stream:
    choice = chunk.choices[0]
    if choice.delta.content:
        print(choice.delta.content, end="", flush=True)
    if choice.finish_reason is not None:
        print(f"\n[Done: {choice.finish_reason}]")
```

---

## Reassembling the Full Response

If you need the complete text after streaming (e.g., for logging):

```python
full_text = ""
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        full_text += delta.content

print(full_text)
```

---

## Token Usage with Streaming

By default, token usage is **not** included in streamed responses (it would only be available in the last chunk). To include it, pass `stream_options={"include_usage": True}`:

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi!"}],
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    if chunk.usage:
        print(f"Total tokens: {chunk.usage.total_tokens}")
```

---

## Streaming in Web Applications

The OpenAI SDK returns tokens over HTTP using **Server-Sent Events (SSE)**. In a web app, you forward these events to the browser:

### FastAPI Example (Python)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI

app = FastAPI()
client = OpenAI()

@app.get("/chat")
async def chat(message: str):
    def generate():
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(generate(), media_type="text/plain")
```

### Browser (JavaScript)

```javascript
const response = await fetch('/chat?message=Hello');
const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  document.getElementById('output').textContent += decoder.decode(value);
}
```

---

## Error Handling in Streams

Network errors can occur mid-stream. Always wrap the stream loop in try/except:

```python
try:
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
except openai.APIConnectionError:
    print("\n[Connection lost — partial response above]")
except openai.RateLimitError:
    print("\n[Rate limited — retry after a delay]")
```

---

## Real-World Tips

1. **Always flush output** when streaming to a terminal or UI: `print(text, end="", flush=True)`.
2. **Use streaming for responses >3 sentences** — the UX improvement is noticeable.
3. **Buffer chunks before displaying** if you need to process complete sentences (e.g., for text-to-speech).
4. **Implement reconnection logic** for long streams — network interruptions happen.
5. **Use `stream_options={"include_usage": True}`** if you need token counts without making a second call.
6. **Test non-streaming too** — some features (like `logprobs`) may behave differently with streaming.

---

## See Also

- [Example code](../examples/06_streaming.py)
- [Autoregressive Generation](05_autoregressive_generation.md)
- [OpenAI Streaming Guide](https://platform.openai.com/docs/api-reference/streaming)
