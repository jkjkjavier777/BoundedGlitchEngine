# GlitchBot

A minimal terminal chatbot that talks to [BoundedGlitchGPT](https://github.com/jkjkjavier777/The-BoundedGlitchGPT) — a small, from-scratch, character-level GPT — over its local HTTP API.

GlitchBot itself contains no model code. It's a thin CLI client: it reads your input, sends it to a running BoundedGlitchGPT server, and prints whatever comes back.

```
You (Termux)
     │
     ▼
GlitchBot (this repo)
     │  HTTP POST /generate
     ▼
BoundedGlitchGPT server (127.0.0.1:8000)
     │
     ▼
generated text
```

## Requirements

- [Termux](https://termux.dev/) (or any Node-capable terminal)
- Node.js 18+ (for native `fetch`)
- A running instance of [The-BoundedGlitchGPT](https://github.com/jkjkjavier777/The-BoundedGlitchGPT) server (`python server/server.py`)

## Installation

```bash
git clone https://github.com/jkjkjavier777/GlitchBot.git
cd GlitchBot
npm install
cp .env.example .env
```

## Usage

**1. Start the BoundedGlitchGPT server** (in a separate Termux session):

```bash
cd ~/The-BoundedGlitchGPT
python server/server.py
```

**2. Start GlitchBot:**

```bash
npm start
```

You'll get an interactive prompt. Type a message and press Enter to generate a response. Type `exit` or `quit` to stop.

## Configuration

Set these in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `GLITCHBOT_SERVER_URL` | `http://127.0.0.1:8000` | Base URL of the BoundedGlitchGPT server |
| `GLITCHBOT_MAX_TOKENS` | `100` | Max tokens generated per response |
| `GLITCHBOT_TEMPERATURE` | `0.8` | Sampling temperature |
| `GLITCHBOT_TIMEOUT_MS` | `30000` | Request timeout in milliseconds |

## A note on the model

BoundedGlitchGPT is a tiny, educational, character-level language model — not a general-purpose chatbot. Expect word fragments and structural mimicry of its training data, not coherent conversation. Its tokenizer only recognizes characters seen during training, so prompts containing unfamiliar characters will be rejected by the server with a `400` error listing the offending characters.

## Project structure

```
GlitchBot/
├── bin/
│   └── glitchbot.js     # CLI entry point
├── src/
│   ├── client.js         # HTTP client for the BoundedGlitchGPT API
│   ├── cli.js             # Interactive terminal loop
│   └── config.js         # Env var loading and defaults
└── test/
    └── client.test.js
```

## License

MIT
