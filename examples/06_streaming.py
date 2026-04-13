"""
Example 6: Streaming Responses
================================
With streaming, the API sends tokens to the client as they are generated
rather than waiting for the complete response.

Benefits:
- Perceived latency drops dramatically — users see the first token almost
  immediately instead of waiting for the entire response.
- Enables real-time UI updates (chatbot typing effect).
- Useful for long responses where total generation takes several seconds.

How it works (Server-Sent Events):
- Pass `stream=True` to the API call.
- The return value is an *iterator* of `ChatCompletionChunk` objects.
- Each chunk has a `delta` field containing the incremental text.
- The stream ends when `finish_reason` is set on the last chunk.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/06_streaming.py
"""

import os
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


def basic_streaming() -> None:
    """Stream a response and print each chunk as it arrives."""
    print("=" * 60)
    print("Basic Streaming")
    print("=" * 60)
    print("Prompt: 'Count from 1 to 10, one number per line.'\n")
    print("Response (streamed):")

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 10, one number per line."}],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)

    print("\n")


def streaming_with_reassembly() -> None:
    """
    Collect all streamed chunks and reassemble the full response.
    This pattern is useful when you need the complete text after streaming.
    """
    print("=" * 60)
    print("Streaming with Reassembly")
    print("=" * 60)

    full_text = ""
    chunk_count = 0

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Explain what streaming means in the context of LLM APIs in 3 sentences.",
            }
        ],
        stream=True,
    )

    print("Streaming... ", end="", flush=True)
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            full_text += delta.content
            chunk_count += 1
            print(".", end="", flush=True)

    print(f" done ({chunk_count} chunks)\n")
    print(f"Full response:\n{full_text}\n")


def streaming_latency_comparison() -> None:
    """
    Compare time-to-first-token (TTFT) for streaming vs. non-streaming.

    TTFT is the key metric for perceived responsiveness.
    """
    print("=" * 60)
    print("Latency Comparison: Streaming vs. Non-Streaming")
    print("=" * 60)

    prompt = "Write a short poem (4 lines) about machine learning."

    # --- Non-streaming ---
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    total_time_non_stream = time.perf_counter() - t0
    print(f"Non-streaming total latency : {total_time_non_stream:.2f}s")
    print(f"  (user waits the full {total_time_non_stream:.2f}s before seeing any text)\n")

    # --- Streaming ---
    t0 = time.perf_counter()
    ttft = None
    full_text = ""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0  # time to first token
            full_text += delta.content
    total_time_stream = time.perf_counter() - t0

    print(f"Streaming time-to-first-token: {ttft:.2f}s  ← user sees text this fast")
    print(f"Streaming total latency       : {total_time_stream:.2f}s\n")
    print(f"Poem:\n{full_text}\n")


def streaming_with_finish_reason() -> None:
    """
    Show how to detect the end of a stream via finish_reason.

    finish_reason values:
      'stop'   – natural end of generation
      'length' – max_tokens reached
      None     – stream not yet finished
    """
    print("=" * 60)
    print("Detecting End of Stream via finish_reason")
    print("=" * 60)

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say exactly: 'Stream complete.'"}],
        stream=True,
        max_tokens=20,
    )

    finish_reason = None
    text = ""
    for chunk in stream:
        choice = chunk.choices[0]
        if choice.delta.content:
            text += choice.delta.content
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason

    print(f"Generated text  : {text!r}")
    print(f"Finish reason   : {finish_reason}")
    print()


def streaming_typewriter_effect() -> None:
    """
    Simulate a typewriter / chatbot UI effect by adding a small delay
    between printed characters.
    """
    print("=" * 60)
    print("Typewriter Effect Demo")
    print("=" * 60)
    print("(simulating a chatbot UI — 10ms delay per character)\n")

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Give me a motivational quote in one sentence."}
        ],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            for char in delta.content:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.01)  # 10 ms per character

    print("\n")


if __name__ == "__main__":
    basic_streaming()
    streaming_with_reassembly()
    streaming_latency_comparison()
    streaming_with_finish_reason()
    streaming_typewriter_effect()
