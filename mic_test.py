"""
One-time calibration: cycles through real (non-virtual) input devices,
recording briefly from each, and reports which ones actually pick up sound.
Keep talking continuously the whole time it's running.
"""

import time

import numpy as np
import sounddevice as sd

SECONDS_PER_DEVICE = 1.5


def main() -> None:
    devices = sd.query_devices()
    candidates = [
        (i, d["name"])
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0 and "virtual" not in d["name"].lower() and "cable" not in d["name"].lower()
    ]

    print(f"Testing {len(candidates)} devices, {SECONDS_PER_DEVICE}s each.")
    print("Start talking now and keep talking continuously until this finishes.\n")
    time.sleep(1.5)

    results = []
    for index, name in candidates:
        try:
            recording = sd.rec(
                int(SECONDS_PER_DEVICE * 44100),
                samplerate=44100,
                channels=1,
                dtype="int16",
                device=index,
            )
            sd.wait()
            peak = int(np.abs(recording).max())
        except Exception as e:
            peak = -1
            print(f"  device {index} ({name}): ERROR - {e}")
            continue
        results.append((peak, index, name))
        print(f"  device {index} ({name}): peak level {peak}")

    print()
    results.sort(reverse=True)
    print("Best candidates (highest peak level first):")
    for peak, index, name in results[:5]:
        print(f"  device {index}: {name}  (peak {peak})")

    if results and results[0][0] > 500:
        print(f"\nLikely your real microphone: device {results[0][1]} -- {results[0][2]}")
    else:
        print("\nNo device picked up clear sound -- try running again and speak louder/closer to the mic.")


if __name__ == "__main__":
    main()
