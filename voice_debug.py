"""
Diagnostic script: prints every mic audio level and every message received
from Deepgram Flux, with nothing filtered out, and writes everything to
voice_debug.log too. Run this and speak normally for ~15 seconds, then
Ctrl+C (or let it finish on its own).
"""

import os
import sys
import threading
import time
import traceback

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from deepgram import DeepgramClient
from deepgram.core.events import EventType

load_dotenv()

SAMPLE_RATE = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * 80 / 1000)
_input_device_env = os.environ.get("INPUT_DEVICE_INDEX")
INPUT_DEVICE_INDEX = int(_input_device_env) if _input_device_env else None
EOT_THRESHOLD = float(os.environ.get("DEEPGRAM_EOT_THRESHOLD", "0.5"))
EOT_TIMEOUT_MS = int(os.environ.get("DEEPGRAM_EOT_TIMEOUT_MS", "3000"))

LOG_PATH = os.path.join(os.path.dirname(__file__), "voice_debug.log")
_log_file = open(LOG_PATH, "w", encoding="utf-8")


def log(*args) -> None:
    msg = " ".join(str(a) for a in args)
    print(msg)
    _log_file.write(msg + "\n")
    _log_file.flush()


audio_callback_count = 0
bytes_sent = 0
listener_crash = None


def main() -> None:
    log("INPUT_DEVICE_INDEX in use:", INPUT_DEVICE_INDEX)
    client = DeepgramClient(api_key=os.environ["DEEPGRAM_API_KEY"])

    with client.listen.v2.connect(
        model="flux-general-en",
        encoding="linear16",
        sample_rate=SAMPLE_RATE,
        eot_threshold=EOT_THRESHOLD,
        eot_timeout_ms=EOT_TIMEOUT_MS,
    ) as stt:
        log("Flux WebSocket opened.")

        def on_message(message) -> None:
            log(f"[FLUX MESSAGE] type={getattr(message, 'type', '?')!r} full={message!r}")

        def on_error(err) -> None:
            log(f"[FLUX ERROR] {err!r}")

        def on_close(*a) -> None:
            log(f"[FLUX CLOSED] {a!r}")

        stt.on(EventType.MESSAGE, on_message)
        stt.on(EventType.ERROR, on_error)
        stt.on(EventType.CLOSE, on_close)

        def listen_wrapper() -> None:
            global listener_crash
            try:
                stt.start_listening()
                log("[LISTENER] start_listening() returned normally (this is unexpected mid-call)")
            except Exception:
                listener_crash = traceback.format_exc()
                log("[LISTENER CRASHED]\n" + listener_crash)

        listen_thread = threading.Thread(target=listen_wrapper, daemon=True)
        listen_thread.start()
        log("Listener thread started, alive:", listen_thread.is_alive())

        def on_mic_audio(indata, frames, time_info, status) -> None:
            global audio_callback_count, bytes_sent
            audio_callback_count += 1
            if status:
                log("[MIC STATUS]", status)
            peak = int(np.abs(indata).max())
            data = indata.tobytes()
            try:
                stt.send_media(data)
                bytes_sent += len(data)
            except Exception as e:
                log("[SEND ERROR]", type(e).__name__, str(e))
            if audio_callback_count % 10 == 0:
                log(f"[MIC] callback #{audio_callback_count}, peak={peak}, total_bytes_sent={bytes_sent}")

        log("Opening mic stream... speak now for ~15 seconds.")
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SAMPLES,
                device=INPUT_DEVICE_INDEX,
                callback=on_mic_audio,
            ):
                for _ in range(15):
                    time.sleep(1)
                    if not listen_thread.is_alive():
                        log("[WARNING] listener thread died mid-test")
                        break
        except KeyboardInterrupt:
            pass
        except Exception:
            log("[INPUT STREAM ERROR]\n" + traceback.format_exc())

    log()
    log("=== SUMMARY ===")
    log("mic callbacks fired:", audio_callback_count)
    log("total bytes sent to Flux:", bytes_sent)
    log("listener thread still alive:", listen_thread.is_alive())
    if listener_crash:
        log("listener crash reason:\n" + listener_crash)
    log()
    log(f"Full log also saved to: {LOG_PATH}")
    _log_file.close()


if __name__ == "__main__":
    main()
