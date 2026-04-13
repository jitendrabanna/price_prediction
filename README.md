# AI Engineering Core Concepts

A beginner-friendly learning repository for core AI engineering concepts. This repository includes clear explanations, runnable Python examples, and notes for real-world usage.

---

## 📚 Topics Covered

| # | Topic | Example | Notes |
|---|-------|---------|-------|
| 1 | Chat Completions API | [examples/01_chat_completions.py](examples/01_chat_completions.py) | [notes/01_chat_completions.md](notes/01_chat_completions.md) |
| 2 | Statelessness of LLMs | [examples/02_statelessness.py](examples/02_statelessness.py) | [notes/02_statelessness.md](notes/02_statelessness.md) |
| 3 | Context Window & Tokens | [examples/03_context_window_tokens.py](examples/03_context_window_tokens.py) | [notes/03_context_window_tokens.md](notes/03_context_window_tokens.md) |
| 4 | Tokenization & Mismatch | [examples/04_tokenization.py](examples/04_tokenization.py) | [notes/04_tokenization.md](notes/04_tokenization.md) |
| 5 | Autoregressive Generation | [examples/05_autoregressive_generation.py](examples/05_autoregressive_generation.py) | [notes/05_autoregressive_generation.md](notes/05_autoregressive_generation.md) |
| 6 | Streaming Responses | [examples/06_streaming.py](examples/06_streaming.py) | [notes/06_streaming.md](notes/06_streaming.md) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)

### Installation

```bash
# Clone the repository
git clone https://github.com/jitendrabanna/price_prediction.git
cd price_prediction

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set your OpenAI API key as an environment variable before running the examples:

```bash
export OPENAI_API_KEY="sk-..."
```

Or create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

### Running the Examples

Each example is a self-contained Python script:

```bash
python examples/01_chat_completions.py
python examples/02_statelessness.py
python examples/03_context_window_tokens.py
python examples/04_tokenization.py
python examples/05_autoregressive_generation.py
python examples/06_streaming.py
```

---

## 📂 Repository Structure

```
.
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── examples/                        # Runnable Python examples
│   ├── 01_chat_completions.py       # Chat Completions API
│   ├── 02_statelessness.py          # Statelessness of LLMs
│   ├── 03_context_window_tokens.py  # Context window and tokens
│   ├── 04_tokenization.py           # Tokenization and mismatch
│   ├── 05_autoregressive_generation.py  # Autoregressive generation
│   └── 06_streaming.py              # Streaming responses
└── notes/                           # Conceptual explanations (Markdown)
    ├── 01_chat_completions.md
    ├── 02_statelessness.md
    ├── 03_context_window_tokens.md
    ├── 04_tokenization.md
    ├── 05_autoregressive_generation.md
    └── 06_streaming.md
```

---

## 💡 Concept Summaries

### 1. Chat Completions API
The Chat Completions API is the primary way to interact with GPT-style models. You send a list of messages (with `system`, `user`, and `assistant` roles) and receive a model-generated reply.

### 2. Statelessness of LLMs
LLMs have no built-in memory — every API call starts fresh. To maintain conversation history, you must explicitly include all prior messages in each request.

### 3. Context Window & Tokens
Every model has a maximum context window (e.g., 128k tokens for GPT-4o). Both your input and the model's output count against this limit. Exceeding it causes errors or truncation.

### 4. Tokenization & Tokenizer Mismatch
Text is split into tokens (not characters or words) before being processed. Different models use different tokenizers, so the same text may produce different token counts — this is important when calculating costs and context limits.

### 5. Autoregressive Generation
LLMs generate text one token at a time, where each new token is conditioned on all previously generated tokens. This is why generation is sequential and why changing one part of a prompt can cascade to change the entire output.

### 6. Streaming Responses
Instead of waiting for the full response, streaming sends tokens to the client as soon as they are generated — giving a faster, more interactive user experience.

---

## 📖 Further Reading

- [OpenAI API Docs](https://platform.openai.com/docs)
- [Tiktoken (OpenAI tokenizer)](https://github.com/openai/tiktoken)
- [OpenAI Cookbook](https://cookbook.openai.com)
- [Anthropic Claude Docs](https://docs.anthropic.com)

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for new topics, improved examples, or bug fixes.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
