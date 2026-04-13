"""
Example 4: Tokenization & Tokenizer Mismatch
=============================================
Tokenization is the process of splitting text into sub-word units called
*tokens* before they are fed into an LLM.

Key facts:
- OpenAI uses the *tiktoken* library (BPE-based tokenizer).
- Different models use different tokenizers (cl100k_base, o200k_base, …).
- The SAME text can produce DIFFERENT token counts across models/providers.
- This matters for: cost estimation, context-window management, and
  understanding model behaviour on unusual inputs.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/04_tokenization.py
"""

import tiktoken


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize(text: str, encoding_name: str) -> tuple[list[int], list[str]]:
    """Return (token_ids, decoded_tokens) for `text` using the given encoding."""
    enc = tiktoken.get_encoding(encoding_name)
    token_ids = enc.encode(text)
    decoded = [enc.decode([tid]) for tid in token_ids]
    return token_ids, decoded


def print_tokenization(text: str, encoding_name: str) -> None:
    """Pretty-print the tokenization of a string."""
    ids, tokens = tokenize(text, encoding_name)
    print(f"\nEncoding : {encoding_name}")
    print(f"Text     : {text!r}")
    print(f"# Tokens : {len(ids)}")
    print(f"Tokens   : {tokens}")
    print(f"Token IDs: {ids}")


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def show_basic_tokenization() -> None:
    """Tokenise a few strings with the most common OpenAI encoding."""
    print("=" * 60)
    print("Basic Tokenization (cl100k_base — GPT-4 / GPT-3.5-turbo)")
    print("=" * 60)

    samples = [
        "Hello, world!",
        "Artificial intelligence",
        "ChatGPT",
        "The quick brown fox",
        "1234567890",
        "hello",          # all lowercase
        "Hello",          # capitalised — different tokens!
        "HELLO",          # all caps
    ]

    for text in samples:
        print_tokenization(text, "cl100k_base")


def show_tokenizer_mismatch() -> None:
    """
    Compare token counts for the same text across different encodings.

    cl100k_base  — used by gpt-3.5-turbo and gpt-4
    o200k_base   — used by gpt-4o and gpt-4o-mini
    p50k_base    — used by older models (text-davinci-003 etc.)
    """
    print("\n" + "=" * 60)
    print("Tokenizer Mismatch Across Encodings")
    print("=" * 60)

    encodings = ["cl100k_base", "o200k_base", "p50k_base"]

    samples = [
        "The price is $99.99.",
        "Привет мир",           # Russian: "Hello world"
        "こんにちは世界",           # Japanese: "Hello world"
        "def fibonacci(n):",
        "aaaaaaaaaa",           # repeated characters
    ]

    header = f"{'Text':<35}" + "".join(f"{enc:<18}" for enc in encodings)
    print(header)
    print("-" * len(header))

    for text in samples:
        row = f"{text!r:<35}"
        for enc_name in encodings:
            enc = tiktoken.get_encoding(enc_name)
            count = len(enc.encode(text))
            row += f"{count:<18}"
        print(row)

    print(
        "\n⚠️  The same text produces different token counts across encodings!"
        "\n   Always use the tokenizer that matches your target model."
    )


def show_special_tokens() -> None:
    """
    Special tokens have reserved IDs and are used for control purposes.
    Examples: <|endoftext|>, <|im_start|>, <|im_end|>
    """
    print("\n" + "=" * 60)
    print("Special Tokens")
    print("=" * 60)

    enc = tiktoken.get_encoding("cl100k_base")

    special_token = "<|endoftext|>"
    # Special tokens are NOT encoded by default — you must allow them explicitly
    ids_with = enc.encode(special_token, allowed_special={special_token})
    ids_without = enc.encode(special_token, allowed_special=set())

    print(f"Text: {special_token!r}")
    print(f"  Encoded WITH  allowed_special : {ids_with}  ({len(ids_with)} token(s))")
    print(f"  Encoded WITHOUT allowed_special: {ids_without} ({len(ids_without)} token(s))")
    print("  Note: without permission, the special token is split into ordinary tokens.")


def show_byte_fallback() -> None:
    """
    Tiktoken uses byte-level fallback for characters outside the vocabulary.
    Rare Unicode characters can become many tokens.
    """
    print("\n" + "=" * 60)
    print("Byte-Level Fallback (rare Unicode)")
    print("=" * 60)

    enc = tiktoken.get_encoding("cl100k_base")
    rare_chars = [
        "🦄",   # unicorn emoji
        "𒀀",   # ancient cuneiform sign
        "⁉️",   # interrobang
    ]

    for ch in rare_chars:
        ids = enc.encode(ch)
        tokens = [enc.decode([tid]) for tid in ids]
        print(f"  {ch!r:10s} → {len(ids)} token(s): {tokens}")


def decode_round_trip() -> None:
    """Verify that encoding → decoding is lossless."""
    print("\n" + "=" * 60)
    print("Encoding / Decoding Round Trip")
    print("=" * 60)

    enc = tiktoken.get_encoding("cl100k_base")
    original = "Round-trip test: 123 tokens! 🚀"
    token_ids = enc.encode(original)
    decoded = enc.decode(token_ids)

    print(f"Original : {original!r}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded  : {decoded!r}")
    print(f"Match    : {'✅' if original == decoded else '❌'}")


if __name__ == "__main__":
    show_basic_tokenization()
    show_tokenizer_mismatch()
    show_special_tokens()
    show_byte_fallback()
    decode_round_trip()
