"""
Phase 2: voice pipeline over your local microphone/speakers.

Deepgram Flux handles speech-to-text AND turn detection in one model --
StartOfTurn/EndOfTurn events tell us when you start and stop talking, so we
don't have to hand-roll voice-activity detection. The same OrderAgent brain
from Phase 1 processes the transcript unchanged. Deepgram Aura-2 speaks the
reply back. If you start talking while the agent is still speaking
(StartOfTurn), playback is cut off immediately -- natural barge-in.
"""

import datetime
import os
import queue
import threading

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.speak.v1.types.speak_v1text import SpeakV1Text

from order_agent import OrderAgent
from text_utils import strip_markdown_for_speech

load_dotenv()

# Quiet diagnostic log -- every Flux message gets written here (not printed)
# so a future issue (like a spurious transcript) can be traced after the
# fact without cluttering what you actually hear/see during the call.
LOG_PATH = os.path.join(os.path.dirname(__file__), "voice_main.log")


def _log(*args) -> None:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] " + " ".join(str(a) for a in args) + "\n")


SAMPLE_RATE = 16000
CHUNK_MS = 80
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # ~2560 bytes/chunk, Deepgram's recommended size for Flux

STT_MODEL = "flux-general-en"
TTS_MODEL = os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")

# Default eot_threshold=0.7 wasn't firing reliably against this setup's audio
# (confidence approached but kept dropping below it). Lowered per Deepgram's
# own tuning guidance for unreliable detection. eot_timeout_ms is a hard
# fallback so a turn ends within a few seconds no matter what.
EOT_THRESHOLD = float(os.environ.get("DEEPGRAM_EOT_THRESHOLD", "0.5"))
EOT_TIMEOUT_MS = int(os.environ.get("DEEPGRAM_EOT_TIMEOUT_MS", "3000"))

# Windows systems often default to a virtual-cable input with no real audio
# flowing into it. Run mic_test.py to find your real microphone's device
# number, then set it here (or leave unset to use the system default).
_input_device_env = os.environ.get("INPUT_DEVICE_INDEX")
INPUT_DEVICE_INDEX = int(_input_device_env) if _input_device_env else None


class VoiceCall:
    def __init__(self):
        self.client = DeepgramClient(api_key=os.environ["DEEPGRAM_API_KEY"])
        self.agent = OrderAgent()
        self.turns = queue.Queue()
        self.speaking = threading.Event()
        self.interrupt = threading.Event()
        self.stt = None

    def run(self) -> None:
        print("=" * 60)
        print(f"{self.agent.menu['restaurant_name']} -- Voice Order Line (Phase 2)")
        print("=" * 60)
        print("Speak into your microphone. Press Ctrl+C to end the call.\n")

        _log("=" * 20, "CALL STARTED", "=" * 20)
        _log(f"eot_threshold={EOT_THRESHOLD} eot_timeout_ms={EOT_TIMEOUT_MS} input_device={INPUT_DEVICE_INDEX}")

        with self.client.listen.v2.connect(
            model=STT_MODEL,
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            eot_threshold=EOT_THRESHOLD,
            eot_timeout_ms=EOT_TIMEOUT_MS,
        ) as stt:
            self.stt = stt
            stt.on(EventType.MESSAGE, self._on_stt_message)
            stt.on(EventType.ERROR, lambda e: print("[STT error]", e))
            threading.Thread(target=stt.start_listening, daemon=True).start()

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SAMPLES,
                device=INPUT_DEVICE_INDEX,
                callback=self._on_mic_audio,
            ):
                self._speak(self.agent.greeting())
                try:
                    while True:
                        transcript = self.turns.get()
                        self._handle_turn(transcript)
                except KeyboardInterrupt:
                    print("\nCall ended.")

    def _on_mic_audio(self, indata, frames, time_info, status) -> None:
        if status:
            _log("[MIC STATUS]", status)
        if self.stt is not None:
            self.stt.send_media(indata.tobytes())

    def _on_stt_message(self, message) -> None:
        # Flux (listen.v2) messages arrive as plain dicts, not objects with
        # attributes -- unlike TTS messages below, which are real objects.
        # Handle both shapes defensively in case the SDK changes this.
        get = message.get if isinstance(message, dict) else lambda k, d=None: getattr(message, k, d)

        if get("type") != "TurnInfo":
            return
        event = get("event")
        # Log every TurnInfo (not just the ones we act on) so a future
        # spurious transcript can be traced back to its confidence/timing.
        _log(
            f"event={event} turn={get('turn_index')} transcript={get('transcript')!r} "
            f"eot_confidence={get('end_of_turn_confidence')} "
            f"window={get('audio_window_start')}-{get('audio_window_end')} "
            f"speaking={self.speaking.is_set()}"
        )
        if event == "StartOfTurn" and self.speaking.is_set():
            self.interrupt.set()
            _log("-> barge-in: interrupting playback")
        elif event == "EndOfTurn":
            transcript = (get("transcript") or "").strip()
            if transcript:
                self.turns.put(transcript)
                _log(f"-> queued as customer turn: {transcript!r}")

    def _handle_turn(self, transcript: str) -> None:
        print(f"Customer: {transcript}")
        _log(f"HANDLING TURN: {transcript!r}")
        reply = self.agent.process_message(transcript)
        print(f"Agent: {reply}")
        _log(f"REPLY: {reply!r}")
        self._speak(reply)
        if self.agent.order_finalized:
            print("\n[Order saved to orders.db]\n")
            _log("ORDER FINALIZED")

    def _speak(self, text: str) -> None:
        text = strip_markdown_for_speech(text)
        self.interrupt.clear()
        self.speaking.set()
        audio_q: queue.Queue = queue.Queue()

        try:
            with self.client.speak.v1.connect(
                model=TTS_MODEL, encoding="linear16", sample_rate=SAMPLE_RATE
            ) as tts:
                def on_tts_message(message) -> None:
                    if isinstance(message, (bytes, bytearray)):
                        audio_q.put(bytes(message))
                    elif getattr(message, "type", None) == "Flushed":
                        audio_q.put(None)  # sentinel: synthesis complete

                tts.on(EventType.MESSAGE, on_tts_message)
                threading.Thread(target=tts.start_listening, daemon=True).start()
                tts.send_text(SpeakV1Text(text=text))
                tts.send_flush()

                with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as out:
                    while True:
                        if self.interrupt.is_set():
                            # Tell the server to stop generating right away, rather
                            # than waiting for it to finish the whole reply -- this
                            # is what was causing the next turn to stall for several
                            # seconds after a barge-in.
                            try:
                                tts.send_close()
                            except Exception as e:
                                _log("[TTS send_close error on interrupt]", type(e).__name__, str(e))
                            break
                        # Short timeout instead of a plain blocking get() so we
                        # keep re-checking the interrupt flag even if audio
                        # delivery stalls, instead of only at chunk boundaries.
                        try:
                            chunk = audio_q.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if chunk is None:
                            break
                        out.write(np.frombuffer(chunk, dtype=np.int16))
        finally:
            self.speaking.clear()


if __name__ == "__main__":
    VoiceCall().run()
