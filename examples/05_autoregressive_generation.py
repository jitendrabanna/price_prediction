"""
Example 5: Autoregressive Generation
======================================
LLMs generate text *one token at a time*.  At each step the model:
  1. Receives all tokens generated so far (plus the original prompt).
  2. Produces a probability distribution over the entire vocabulary.
  3. Samples (or greedily picks) the next token.
  4. Appends that token and repeats until a stop condition is reached.

This sequential, left-to-right generation is called *autoregressive*.

Key implications:
- Generation is O(n) in the number of output tokens — it cannot be
  fully parallelised across output positions.
- Earlier tokens heavily influence later ones (context dependency).
- `temperature` and `top_p` control how the model samples from the
  probability distribution at each step.
- `max_tokens` limits how many autoregressive steps are taken.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/05_autoregressive_generation.py
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------

def compare_temperature() -> None:
    """
    Show how temperature affects the diversity of completions.

    temperature=0.0  → almost deterministic (greedy-ish)
    temperature=1.0  → moderate randomness
    temperature=2.0  → very random / creative (often incoherent at extremes)
    """
    prompt = "Continue this sentence in exactly 10 words: 'The robot opened the door and'"

    print("=" * 60)
    print("Effect of Temperature on Autoregressive Sampling")
    print("=" * 60)
    print(f"Prompt: {prompt!r}\n")

    for temp in [0.0, 0.7, 1.5]:
        # Run twice at the same temperature to show variance
        results = []
        for _ in range(2):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=20,
            )
            results.append(response.choices[0].message.content.strip())

        print(f"temperature={temp}")
        for i, r in enumerate(results, 1):
            print(f"  Run {i}: {r!r}")
        print()


def compare_top_p() -> None:
    """
    Show how top_p (nucleus sampling) affects generation.

    top_p=0.1  → only the top 10% probability mass is considered
    top_p=0.9  → top 90% probability mass (more diverse)
    top_p=1.0  → full vocabulary (same as no nucleus filtering)
    """
    prompt = "Name a random animal:"

    print("=" * 60)
    print("Effect of top_p (Nucleus Sampling)")
    print("=" * 60)
    print(f"Prompt: {prompt!r}\n")

    for top_p in [0.1, 0.5, 1.0]:
        results = []
        for _ in range(3):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                top_p=top_p,
                max_tokens=10,
            )
            results.append(response.choices[0].message.content.strip())

        print(f"top_p={top_p} → {results}")
    print()


def show_max_tokens_truncation() -> None:
    """
    Demonstrate that max_tokens stops generation mid-sentence.

    finish_reason will be 'length' (truncated) rather than 'stop' (natural end).
    """
    print("=" * 60)
    print("max_tokens Truncation")
    print("=" * 60)

    for max_tok in [5, 20, 100]:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Explain autoregressive generation in detail."}
            ],
            max_tokens=max_tok,
        )
        choice = response.choices[0]
        print(
            f"max_tokens={max_tok:3d} | finish={choice.finish_reason:6s} | "
            f"text: {choice.message.content!r}"
        )
    print()


def measure_generation_latency() -> None:
    """
    Measure wall-clock latency for different output lengths to illustrate
    that generation time scales with the number of output tokens.
    """
    print("=" * 60)
    print("Generation Latency vs. Output Length")
    print("=" * 60)
    print("(Latency increases with more output tokens because generation is sequential)\n")

    for max_tok in [10, 50, 200]:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Write a story about a robot."}],
            max_tokens=max_tok,
            temperature=0.7,
        )
        elapsed = time.perf_counter() - start
        actual_tokens = response.usage.completion_tokens
        print(
            f"max_tokens={max_tok:3d} | actual_completion_tokens={actual_tokens:3d} | "
            f"latency={elapsed:.2f}s"
        )
    print()


def show_stop_sequences() -> None:
    """
    Stop sequences tell the model to halt generation at a specific string.
    This is useful for structured output formats.
    """
    print("=" * 60)
    print("Stop Sequences")
    print("=" * 60)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "List three fruits, one per line. "
                    "After each fruit write ' [END]'."
                ),
            }
        ],
        stop=["[END]"],
        max_tokens=50,
        temperature=0.0,
    )

    print(f"finish_reason : {response.choices[0].finish_reason}")
    print(f"output        : {response.choices[0].message.content!r}")
    print("Note: generation stopped when it was about to produce '[END]'.")


if __name__ == "__main__":
    compare_temperature()
    compare_top_p()
    show_max_tokens_truncation()
    measure_generation_latency()
    show_stop_sequences()
