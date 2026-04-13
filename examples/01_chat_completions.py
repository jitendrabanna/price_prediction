"""
Example 1: Chat Completions API
================================
Demonstrates how to call the OpenAI Chat Completions API.

Key concepts:
- The `messages` list drives the conversation using three roles:
    system   – sets the assistant's behaviour / persona
    user     – the human turn
    assistant – a previous model reply (used to build history)
- `model` selects which LLM to use.
- The response object contains the generated text in
  response.choices[0].message.content

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/01_chat_completions.py
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load OPENAI_API_KEY from a .env file if present
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def basic_completion(user_message: str) -> str:
    """Send a single user message and return the assistant's reply."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers concisely.",
            },
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def multi_turn_completion() -> None:
    """
    Simulate a short multi-turn conversation.

    Notice how we manually append each turn to the `messages` list so
    the model has context from previous exchanges.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable tutor specialising in AI.",
        }
    ]

    turns = [
        "What is a large language model (LLM)?",
        "How does it differ from a traditional search engine?",
    ]

    for user_input in turns:
        print(f"\n👤 User: {user_input}")
        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        assistant_reply = response.choices[0].message.content
        print(f"🤖 Assistant: {assistant_reply}")

        # Append the assistant reply so the next turn has full context
        messages.append({"role": "assistant", "content": assistant_reply})


def inspect_response_object() -> None:
    """Show the most useful fields of the response object."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Hello, AI World!' exactly."}],
    )

    print("\n--- Response Object Fields ---")
    print(f"Model used      : {response.model}")
    print(f"Finish reason   : {response.choices[0].finish_reason}")
    print(f"Prompt tokens   : {response.usage.prompt_tokens}")
    print(f"Completion tokens: {response.usage.completion_tokens}")
    print(f"Total tokens    : {response.usage.total_tokens}")
    print(f"Content         : {response.choices[0].message.content}")


if __name__ == "__main__":
    print("=== Basic Completion ===")
    reply = basic_completion("Explain AI in one sentence.")
    print(f"Reply: {reply}")

    print("\n=== Multi-Turn Conversation ===")
    multi_turn_completion()

    print("\n=== Response Object Inspection ===")
    inspect_response_object()
