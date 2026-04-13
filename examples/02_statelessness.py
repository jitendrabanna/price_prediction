"""
Example 2: Statelessness of LLMs
==================================
LLMs have NO built-in memory between API calls.
Every call is completely independent — the model knows nothing about
previous calls unless you explicitly include that history in `messages`.

This example shows:
1. The "amnesia" problem – calling the API twice without history.
2. The fix – manually passing the full conversation history.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/02_statelessness.py
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


def demonstrate_amnesia() -> None:
    """
    Show that the model does NOT remember previous calls.

    We first tell the model our name, then immediately ask it in a
    *separate* API call — it cannot recall what we said.
    """
    print("--- Without history (stateless / amnesia) ---")

    # Call 1: introduce ourselves
    response1 = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi! My name is Alex."}],
    )
    print(f"Call 1 reply: {response1.choices[0].message.content}")

    # Call 2: ask it to recall (completely new API call — no context)
    response2 = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is my name?"}],
    )
    print(f"Call 2 reply (no context): {response2.choices[0].message.content}")
    print("⚠️  The model has no memory of Call 1!")


def demonstrate_manual_history() -> None:
    """
    Fix the amnesia problem by including history in every request.

    We build a `messages` list and append each turn ourselves before
    sending the next request.
    """
    print("\n--- With manual history (stateless + history management) ---")

    messages = []

    # Turn 1: introduce ourselves
    user_msg_1 = "Hi! My name is Alex."
    messages.append({"role": "user", "content": user_msg_1})

    response1 = client.chat.completions.create(model=MODEL, messages=messages)
    assistant_reply_1 = response1.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_reply_1})

    print(f"Turn 1 — User    : {user_msg_1}")
    print(f"Turn 1 — Assistant: {assistant_reply_1}")

    # Turn 2: ask for our name (history is included this time)
    user_msg_2 = "What is my name?"
    messages.append({"role": "user", "content": user_msg_2})

    response2 = client.chat.completions.create(model=MODEL, messages=messages)
    assistant_reply_2 = response2.choices[0].message.content

    print(f"\nTurn 2 — User    : {user_msg_2}")
    print(f"Turn 2 — Assistant: {assistant_reply_2}")
    print("✅  The model correctly recalled the name because history was passed.")


def show_context_as_state() -> None:
    """
    Illustrate that the full messages list IS the model's 'state'.
    Print the messages list before each call so the pattern is clear.
    """
    print("\n--- Inspecting messages list as state ---")

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
    ]

    conversations = [
        "My favourite colour is blue.",
        "What is my favourite colour?",
        "Now forget that. My favourite colour is actually green.",
        "So what is my favourite colour now?",
    ]

    for user_input in conversations:
        messages.append({"role": "user", "content": user_input})
        print(f"\n[Messages being sent — {len(messages)} items]")

        response = client.chat.completions.create(model=MODEL, messages=messages)
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        print(f"User      : {user_input}")
        print(f"Assistant : {reply}")


if __name__ == "__main__":
    demonstrate_amnesia()
    demonstrate_manual_history()
    show_context_as_state()
