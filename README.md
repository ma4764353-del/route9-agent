# Route9 — AI Phone-Ordering Agent for Restaurants

Route9 answers a restaurant's phone calls, takes the order in natural conversation,
reads it back with the total, and saves it. It can also text back customers whose
call was missed and take their order over SMS.

The agent is built on a Claude (or Gemini) tool-calling "brain", with Deepgram for
speech and Twilio for the phone line.

## What it does

- Takes orders by voice (phone call) or by text (SMS or command line)
- Understands modifiers ("no onions", "extra cheese"), quantity changes and cancellations
- Answers menu questions and refuses items that aren't on the menu
- Reads the order and total back, then saves it to a local SQLite database
- Optionally texts the restaurant staff as soon as an order is confirmed
- Handles interruptions: if the caller talks over the agent, it stops speaking

## How it works

```
Caller ──► Twilio phone number ──► server.py (FastAPI, on Railway or similar)
                                       │  audio over WebSocket (8kHz mulaw)
                                       ▼
                       Deepgram Flux: speech-to-text + turn detection
                                       │  transcript
                                       ▼
                    order_agent.py: the "brain"
                    Claude / Gemini with tools: add_item, remove_item,
                    modify_item, get_menu_info, confirm_order
                                       │  reply text
                                       ▼
                       Deepgram Aura-2: text-to-speech ──► back to the caller
                                       │
                                       ▼
                         orders.db (SQLite): orders + call logs
```

The same brain (`order_agent.py`) powers every channel. Only the input and output change.

The project was built in three phases, and each one still runs on its own:

| Phase | What | Entry point |
|-------|------|-------------|
| 1 | Text-only order bot in the terminal | `main.py` |
| 2 | Voice using your computer's mic and speakers | `voice_main.py` |
| 3 | Real phone calls through Twilio | `server.py` |

There is also a **text-back** mode: when a restaurant forwards its unanswered calls to
a Twilio number, the server hangs up politely and texts the caller. The caller can then
reply with their order and the same brain handles it over SMS.

## Project layout

| File | Purpose |
|------|---------|
| `order_agent.py` | Core logic: menu, pricing, cart, tools, SQLite saving |
| `llm_providers.py` | Swappable LLM backends (Anthropic Claude or Google Gemini) |
| `server.py` | Phone server: Twilio webhooks, audio streaming, text-back SMS |
| `main.py` | Text-based command-line demo |
| `voice_main.py` | Local voice demo (microphone and speakers) |
| `menu.json` | The restaurant's menu, prices, modifiers and hours (sample data) |
| `text_utils.py` | Cleans markdown out of replies before they are spoken |
| `view_log.py` | Prints saved calls and orders |
| `mic_test.py`, `voice_debug.py` | Tools for finding and debugging your microphone |

## Quick start (text demo)

You need Python 3.11+ and an API key from Anthropic or Google Gemini.

```bash
pip install -r requirements.txt
cp .env.example .env        # then open .env and add your key(s)
python main.py              # chat with the agent like a customer
python view_log.py          # see saved calls and orders afterwards
```

## Running with voice or a real phone number

- **Local voice:** add `DEEPGRAM_API_KEY` to `.env`, then run `python voice_main.py`.
  Run `python mic_test.py` first if it can't find your microphone.
- **Real phone calls:** deploy `server.py` somewhere reachable on the internet
  (it reads the `PORT` environment variable, so it works on Railway, Render, etc.).
  Then in the Twilio Console, point your phone number's voice webhook to
  `https://YOUR-HOST/voice`.
- **Text-back mode:** point a second Twilio number's voice webhook to
  `https://YOUR-HOST/textback/voice` and its SMS webhook to `https://YOUR-HOST/textback/sms`.

```bash
python server.py
```

## Configuration

Everything is set through environment variables. Copy `.env.example` to `.env`. The
`.env` file is ignored by git, so your keys are never uploaded.

| Variable | Needed for |
|----------|-----------|
| `LLM_PROVIDER` | `anthropic` (default) or `gemini` |
| `ANTHROPIC_API_KEY` | Using Claude |
| `GEMINI_API_KEY` | Using Gemini |
| `DEEPGRAM_API_KEY` | Voice (local and phone) |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` | Phone calls and SMS |
| `TWILIO_SMS_FROM_NUMBER`, `RESTAURANT_NOTIFY_NUMBER` | Texting the restaurant when an order is confirmed (optional) |
| `TWILIO_TEXTBACK_NUMBER` | Text-back mode |

To use your own restaurant, edit `menu.json`. The included menu is made-up sample data.

## Status

Built and deployed as a working prototype. It is not a finished commercial product, so
expect rough edges, and test it carefully before putting it on a real restaurant line.

## Security note

Never commit your `.env` file or share your Twilio Auth Token. It gives full access to
your Twilio account.

## Contact

Questions or feedback: ma4764353@gmail.com
