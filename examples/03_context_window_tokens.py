"""
Example 3: Context Window & Tokens
====================================
Every model has a fixed context window — the maximum number of tokens it
can process in a single call (input + output combined).

- gpt-4o-mini  : 128,000 tokens (as of 2024)
- gpt-4o       : 128,000 tokens
- gpt-3.5-turbo: 16,385 tokens

Key points:
- Tokens ≠ words.  Roughly 1 token ≈ 4 English characters / ¾ of a word.
- Both prompt tokens AND completion tokens count against the limit.
- Exceeding the context window raises an error (context_length_exceeded).
- Cost is calculated per token, so smaller prompts = lower cost.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/03_context_window_tokens.py
"""

import os
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# Context window limits (tokens) for common models
CONTEXT_WINDOWS = {
    "gpt-4o":        128_000,
    "gpt-4o-mini":   128_000,
    "gpt-3.5-turbo": 16_385,
}


def count_tokens(text: str, model: str = MODEL) -> int:
    """Return the number of tokens in `text` for the given model."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def count_messages_tokens(messages: list[dict], model: str = MODEL) -> int:
    """
    Count the tokens that a messages list will consume when sent to the API.

    OpenAI adds small overhead per message (role + formatting tokens).
    This function mirrors the calculation described in the OpenAI cookbook.
    """
    enc = tiktoken.encoding_for_model(model)
    tokens_per_message = 3  # every message: <|start|>{role}\n{content}<|end|>
    tokens_per_name = 1     # if a name field is present

    total = 0
    for msg in messages:
        total += tokens_per_message
        for key, value in msg.items():
            total += len(enc.encode(value))
            if key == "name":
                total += tokens_per_name

    total += 3  # every reply is primed with <|start|>assistant<|im_start|>
    return total


def show_token_counts() -> None:
    """Demonstrate how different texts produce different token counts."""
    samples = [
        "Hello!",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming every industry.",
        "AI",
        "Pneumonoultramicroscopicsilicovolcanoconiosis",  # long word
    ]

    print("--- Token counts for sample strings ---")
    for text in samples:
        n = count_tokens(text)
        print(f"  {n:3d} tokens | {text!r}")


def check_context_budget(messages: list[dict], max_completion_tokens: int = 500) -> None:
    """
    Check whether a messages list fits within the model's context window
    given a desired completion budget.
    """
    limit = CONTEXT_WINDOWS[MODEL]
    prompt_tokens = count_messages_tokens(messages)
    remaining = limit - prompt_tokens

    print(f"\n--- Context Budget Check ({MODEL}) ---")
    print(f"Context window     : {limit:,} tokens")
    print(f"Prompt tokens      : {prompt_tokens:,} tokens")
    print(f"Available for reply: {remaining:,} tokens")

    if remaining < max_completion_tokens:
        print(
            f"⚠️  Only {remaining} tokens left — requested {max_completion_tokens} "
            f"for completion.  Consider shortening the prompt."
        )
    else:
        print(f"✅  Enough room for a {max_completion_tokens}-token completion.")


def call_with_token_usage() -> None:
    """Make an API call and print how many tokens were actually consumed."""
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What is a neural network? Answer in 2 sentences."},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=150,
    )

    usage = response.usage
    print("\n--- Actual Token Usage from API Response ---")
    print(f"Prompt tokens    : {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens     : {usage.total_tokens}")
    print(f"Reply            : {response.choices[0].message.content}")


def demonstrate_truncation_risk() -> None:
    """
    Show how a growing conversation approaches the context limit.
    (We simulate without actually hitting the limit.)
    """
    print("\n--- Growing Conversation Token Count ---")
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    limit = CONTEXT_WINDOWS[MODEL]

    fictional_turns = [
        ("user", "Tell me a long story about a dragon."),
        ("assistant", "Once upon a time, in a land far away, there lived a great dragon named Ember who guarded an ancient library containing the knowledge of the world..."),
        ("user", "What happened next?"),
        ("assistant", "Ember discovered that a group of scholars sought to steal the rarest book — a tome containing the secret of infinite intelligence..."),
        ("user", "Did the scholars succeed?"),
    ]

    for role, content in fictional_turns:
        messages.append({"role": role, "content": content})
        used = count_messages_tokens(messages)
        pct = used / limit * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  [{bar}] {used:,}/{limit:,} tokens ({pct:.1f}%)")


if __name__ == "__main__":
    show_token_counts()

    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum entanglement in simple terms."},
    ]
    check_context_budget(sample_messages)

    call_with_token_usage()
    demonstrate_truncation_risk()
