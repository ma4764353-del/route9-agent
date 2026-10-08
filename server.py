"""
Phase 3: the server real phone calls actually talk to.

Twilio Media Streams connects here over WebSocket during a live call and
exchanges audio as base64-encoded 8kHz mulaw -- a format Deepgram Flux/Aura-2
accept directly (verified live), so no resampling is needed anywhere. The
STT -> OrderAgent brain -> TTS pipeline is the same pattern as voice_main.py
(Phase 2); only the audio source/sink changed, from local mic/speakers to
Twilio's WebSocket messages.

Twilio's own "clear" event tells it to drop any audio it has buffered for
playback -- our side of barge-in. Combined with our own interrupt flag (which
stops us generating more audio), a customer talking over the agent cuts
speech off on both ends at once.
"""

import asyncio
import base64
import json
import os
import queue
import threading
import time

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from starlette.websockets import WebSocketDisconnect
from fastapi.responses import Response
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.speak.v1.types.speak_v1text import SpeakV1Text

from order_agent import OrderAgent, load_menu
from text_utils import strip_markdown_for_speech

load_dotenv()

app = FastAPI()

SAMPLE_RATE = 8000
STT_MODEL = "flux-general-en"
TTS_MODEL = os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
EOT_THRESHOLD = float(os.environ.get("DEEPGRAM_EOT_THRESHOLD", "0.6"))
EOT_TIMEOUT_MS = int(os.environ.get("DEEPGRAM_EOT_TIMEOUT_MS", "5500"))


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class TwilioCall:
    """One phone call. Owns a background worker thread that runs the same
    STT -> brain -> TTS loop as Phase 2, communicating with the async
    WebSocket handler via thread-safe queues."""

    def __init__(self, loop: asyncio.AbstractEventLoop, outbound: "asyncio.Queue[dict]") -> None:
        self.loop = loop
        # A single coroutine (started by the WebSocket handler) drains this
        # queue and awaits websocket.send_json() one at a time. Starlette's
        # WebSocket doesn't serialize concurrent send() calls, so multiple
        # sends in flight at once (e.g. one per-TTS-chunk task) corrupts the
        # connection -- funneling everything through one queue/consumer
        # keeps sends strictly sequential.
        self.outbound = outbound
        self.client = DeepgramClient(api_key=os.environ["DEEPGRAM_API_KEY"])
        self.agent = OrderAgent()
        self.turns: queue.Queue = queue.Queue()
        self.inbound_audio: queue.Queue = queue.Queue()
        self.speaking = threading.Event()
        self.interrupt = threading.Event()
        self.stop_event = threading.Event()
        self.stt = None
        self.stream_sid: str | None = None
        self._started = threading.Event()  # guards against a duplicate Twilio
        # "start" event spawning a second worker thread on top of the first --
        # two threads racing on the same call would each connect their own
        # Flux/TTS socket and speak their own greeting independently.

    def _send(self, event: dict) -> None:
        self.loop.call_soon_threadsafe(self.outbound.put_nowait, event)

    def feed_audio(self, mulaw_bytes: bytes) -> None:
        # Called from the async WebSocket handler on every inbound "media"
        # event (every ~20ms) -- must stay non-blocking. The actual
        # stt.send_media() call is a blocking network write, so it happens
        # on _audio_sender's own thread instead of the asyncio event loop.
        self.inbound_audio.put(mulaw_bytes)

    def _audio_sender(self) -> None:
        while not self.stop_event.is_set():
            try:
                chunk = self.inbound_audio.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.stt is not None:
                try:
                    self.stt.send_media(chunk)
                except Exception as e:
                    # A transient send failure used to kill this daemon thread
                    # silently -- audio would stop reaching Flux for the rest of
                    # the call with no sign anything was wrong. Log and keep
                    # going instead; if the underlying connection is really
                    # gone, _worker's own Flux connection will surface that.
                    print(f"[audio_sender] send_media FAILED: {type(e).__name__}: {e}", flush=True)

    def start(self) -> None:
        if self._started.is_set():
            # A duplicate "start" event from Twilio (reconnect/retry quirk)
            # would otherwise spawn a second worker thread on the same call --
            # both racing to connect Flux/TTS and speak the greeting
            # independently. Ignore anything past the first.
            print("[handler] duplicate start event ignored", flush=True)
            return
        self._started.set()
        threading.Thread(target=self._worker_wrapper, daemon=True).start()

    def stop(self) -> None:
        self.stop_event.set()
        self.interrupt.set()

    def _worker_wrapper(self) -> None:
        # Exceptions in a daemon thread don't propagate anywhere useful --
        # log them explicitly so a failure here is never silent.
        print("[worker] thread started", flush=True)
        try:
            self._worker()
        except Exception:
            import traceback

            print("[TwilioCall worker CRASHED]\n" + traceback.format_exc(), flush=True)
        print("[worker] thread exiting", flush=True)

    def _worker(self) -> None:
        print("[worker] connecting to Flux...", flush=True)
        with self.client.listen.v2.connect(
            model=STT_MODEL,
            encoding="mulaw",
            sample_rate=SAMPLE_RATE,
            eot_threshold=EOT_THRESHOLD,
            eot_timeout_ms=EOT_TIMEOUT_MS,
        ) as stt:
            self.stt = stt
            stt.on(EventType.MESSAGE, self._on_stt_message)
            threading.Thread(target=stt.start_listening, daemon=True).start()
            threading.Thread(target=self._audio_sender, daemon=True).start()
            print("[worker] Flux connected, speaking greeting...", flush=True)

            self._speak(self.agent.greeting())
            print("[worker] greeting done", flush=True)

            while not self.stop_event.is_set():
                try:
                    transcript = self.turns.get(timeout=0.5)
                except queue.Empty:
                    continue
                self._handle_turn(transcript)

    def _on_stt_message(self, message) -> None:
        get = message.get if isinstance(message, dict) else lambda k, d=None: getattr(message, k, d)
        if get("type") != "TurnInfo":
            return
        event = get("event")
        if event == "StartOfTurn":
            if self.speaking.is_set():
                print("[stt] StartOfTurn while speaking -- triggering barge-in interrupt", flush=True)
                self.interrupt.set()
                self._send({"event": "clear", "streamSid": self.stream_sid})
            else:
                print("[stt] StartOfTurn (agent not speaking, no interrupt needed)", flush=True)
        elif event == "EndOfTurn":
            transcript = (get("transcript") or "").strip()
            print(f"[stt] EndOfTurn: {transcript!r}", flush=True)
            if transcript:
                self.turns.put(transcript)

    def _handle_turn(self, transcript: str) -> None:
        print(f"[turn] sending to LLM: {transcript!r}", flush=True)
        start = time.monotonic()
        reply = self.agent.process_message(transcript)
        print(f"[turn] LLM replied in {time.monotonic() - start:.2f}s: {reply!r}", flush=True)
        self._speak(reply)

    def _speak(self, text: str) -> None:
        text = strip_markdown_for_speech(text)
        print(f"[tts] speaking: {text!r}", flush=True)
        tts_start = time.monotonic()
        self.interrupt.clear()
        self.speaking.set()
        audio_q: queue.Queue = queue.Queue()

        try:
            with self.client.speak.v1.connect(
                model=TTS_MODEL, encoding="mulaw", sample_rate=SAMPLE_RATE
            ) as tts:
                def on_tts_message(message) -> None:
                    if isinstance(message, (bytes, bytearray)):
                        audio_q.put(bytes(message))
                    elif getattr(message, "type", None) == "Flushed":
                        audio_q.put(None)

                tts.on(EventType.MESSAGE, on_tts_message)
                threading.Thread(target=tts.start_listening, daemon=True).start()
                tts.send_text(SpeakV1Text(text=text))
                tts.send_flush()

                while True:
                    if self.interrupt.is_set():
                        try:
                            tts.send_close()
                        except Exception:
                            pass
                        break
                    try:
                        chunk = audio_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if chunk is None:
                        break
                    payload = base64.b64encode(chunk).decode("ascii")
                    self._send({"event": "media", "streamSid": self.stream_sid, "media": {"payload": payload}})
        except Exception as e:
            # A Deepgram TTS connection failure (transient network/API hiccup)
            # used to propagate all the way up and crash the whole worker
            # thread -- silently ending the customer's ability to hear or be
            # heard for the rest of the call. Log it and let the call continue
            # instead; the next turn gets a fresh TTS connection attempt.
            print(f"[tts] FAILED: {type(e).__name__}: {e}", flush=True)
        finally:
            self.speaking.clear()
            print(f"[tts] done in {time.monotonic() - tts_start:.2f}s", flush=True)


@app.get("/")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/voice")
async def voice_webhook(request: Request) -> Response:
    host = request.headers.get("host") or request.url.hostname
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f'<Stream url="wss://{host}/media-stream" />'
        "</Connect></Response>"
    )
    return Response(content=twiml, media_type="text/xml")


# ---------- text-back mode (runs on its own Twilio number) ----------
#
# A restaurant forwards its own line to this number whenever a call goes
# unanswered (conditional call forwarding, set up on the existing phone).
# The webhook answers just long enough to say one line and hang up, then
# texts the caller back. The caller replies with their order and the same
# OrderAgent handles it over SMS. This is a separate code path from /voice
# and /media-stream above -- it shares only the Twilio account and the
# SMS-sending pattern (order_agent.py's _send_restaurant_sms uses the same one).

# Keyed by caller phone number. In-memory only -- resets on restart, so a
# caller who goes quiet for a long stretch and comes back just starts a
# fresh order.
_sms_agents: dict[str, OrderAgent] = {}


def _send_textback_sms(to_number: str, body: str) -> bool:
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_TEXTBACK_NUMBER")
    if not all([account_sid, auth_token, from_number]):
        print("[textback] missing TWILIO_ACCOUNT_SID/AUTH_TOKEN/TEXTBACK_NUMBER -- not sending", flush=True)
        return False
    try:
        from twilio.rest import Client

        Client(account_sid, auth_token).messages.create(body=body, from_=from_number, to=to_number)
        return True
    except Exception as e:
        print(f"[textback] SMS send FAILED to {to_number}: {type(e).__name__}: {e}", flush=True)
        return False


@app.post("/textback/voice")
async def textback_voice_webhook(request: Request) -> Response:
    form = await request.form()
    caller = form.get("From", "")

    restaurant_name = "the restaurant"
    try:
        restaurant_name = load_menu()["restaurant_name"].rstrip(".")
    except Exception as e:
        print(f"[textback] menu lookup for greeting FAILED: {type(e).__name__}: {e}", flush=True)

    if caller:
        body = (
            f"Sorry we missed your call to {restaurant_name}! Reply to this text "
            "with your order and we'll get it started right away."
        )
        _send_textback_sms(caller, body)

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Say>Sorry we missed you. We just texted you to take your order.</Say>"
        "<Hangup/></Response>"
    )
    return Response(content=twiml, media_type="text/xml")


@app.post("/textback/sms")
async def textback_sms_webhook(request: Request) -> Response:
    form = await request.form()
    caller = form.get("From", "")
    text = (form.get("Body") or "").strip()

    if not caller or not text:
        reply = "Sorry, I didn't catch that -- could you send your order again?"
    else:
        agent = _sms_agents.get(caller)
        if agent is None:
            agent = OrderAgent()
            _sms_agents[caller] = agent
        try:
            reply = agent.process_message(text)
        except Exception as e:
            print(f"[textback] process_message FAILED for {caller}: {type(e).__name__}: {e}", flush=True)
            reply = "Sorry, something went wrong on our end -- could you try sending that again?"
        reply = strip_markdown_for_speech(reply)  # same "no literal markdown" cleanup as speech output

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Message>{_xml_escape(reply)}</Message></Response>"
    )
    return Response(content=twiml, media_type="text/xml")


async def _sender_loop(websocket: WebSocket, outbound: "asyncio.Queue[dict]") -> None:
    """The only coroutine allowed to call websocket.send_json() -- keeps all
    outbound sends strictly sequential (see TwilioCall._send)."""
    while True:
        event = await outbound.get()
        if event is None:
            break
        try:
            await websocket.send_json(event)
        except Exception as e:
            print(f"[send FAILED] event={event.get('event')} error={type(e).__name__}: {e}", flush=True)


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    call = None
    sender_task = None
    outbound: "asyncio.Queue[dict]" = asyncio.Queue()

    try:
        loop = asyncio.get_running_loop()
        sender_task = asyncio.create_task(_sender_loop(websocket, outbound))
        call = TwilioCall(loop, outbound)
        print("[handler] TwilioCall constructed OK", flush=True)

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                call.stream_sid = msg["start"]["streamSid"]
                print("[handler] got start event, calling call.start()", flush=True)
                call.start()
            elif event == "media":
                call.feed_audio(base64.b64decode(msg["media"]["payload"]))
            elif event == "stop":
                call.stop()
                break
    except WebSocketDisconnect as e:
        print(f"[handler] WebSocketDisconnect on receive: code={getattr(e, 'code', None)}", flush=True)
    except Exception:
        import traceback

        print("[media_stream handler CRASHED]\n" + traceback.format_exc(), flush=True)
    finally:
        if call is not None:
            call.stop()
        if sender_task is not None:
            # Let the sender drain and exit on its own rather than yanking it
            # mid-write, then explicitly close so the client gets a proper
            # WebSocket close handshake instead of an abrupt TCP drop.
            await outbound.put(None)
            try:
                await asyncio.wait_for(sender_task, timeout=2)
            except (TimeoutError, asyncio.CancelledError):
                sender_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass  # already closed (e.g. caller hung up) -- nothing to do


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
