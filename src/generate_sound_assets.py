"""
RoadWatch AI - Open-Source Audio Asset Generator
------------------------------------------------
Generates platform-agnostic, 100% royalty-free, open-source 16-bit PCM .WAV audio cues
using Python's standard library (wave, math, struct) with zero external audio dependencies.

Outputs:
- assets/sounds/chime_caution.wav: Ascending two-tone melodic chime for CAUTION alerts.
- assets/sounds/chime_critical.wav: Rapid, urgent multi-tone pulse for CRITICAL danger alerts.
"""

import os
import math
import wave
import struct
from typing import List, Tuple


def generate_synthesized_wav(
    output_path: str,
    tones: List[Tuple[float, float, float]],  # List of (frequency_hz, duration_sec, amplitude [0-1])
    sample_rate: int = 44100,
):
    """
    Synthesizes sine wave audio samples with smooth exponential envelope to avoid clicks.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_samples = []

    for freq, duration, amp in tones:
        num_samples = int(sample_rate * duration)
        if freq == 0 or amp == 0:
            # Silence
            total_samples.extend([0.0] * num_samples)
            continue

        for i in range(num_samples):
            t = i / sample_rate
            # Smooth attack and decay envelope to prevent harsh pops/clicks
            # Attack over first 5% of duration, decay over remaining 95%
            attack_len = int(num_samples * 0.08)
            if i < attack_len:
                envelope = i / max(1, attack_len)
            else:
                envelope = math.exp(-3.5 * (i - attack_len) / (num_samples - attack_len))

            sample_val = amp * envelope * math.sin(2.0 * math.pi * freq * t)
            total_samples.append(sample_val)

    # Encode to 16-bit signed PCM integers (-32768 to 32767)
    with wave.open(output_path, "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit (2 bytes per sample)
        wav_file.setframerate(sample_rate)

        raw_bytes = bytearray()
        for s in total_samples:
            clamped = max(-1.0, min(1.0, s))
            int_val = int(clamped * 32767)
            raw_bytes.extend(struct.pack("<h", int_val))

        wav_file.writeframes(raw_bytes)

    print(f"✓ Generated open-source sound: {output_path} ({len(total_samples) / sample_rate:.2f}s)")


def generate_all_sound_assets(target_dir: str = "assets/sounds"):
    """Generates both Caution and Critical open-source audio cues."""
    # 1. Caution Chime: Ascending D5 (587 Hz) -> A5 (880 Hz)
    caution_path = os.path.join(target_dir, "chime_caution.wav")
    caution_tones = [
        (587.33, 0.12, 0.65),  # D5 chime
        (0.0, 0.03, 0.0),      # Micro pause
        (880.00, 0.25, 0.75),  # A5 chime
    ]
    generate_synthesized_wav(caution_path, caution_tones)

    # 2. Critical Alert: Urgent rapid pulsing alert (1046 Hz -> 1318 Hz -> 1046 Hz)
    critical_path = os.path.join(target_dir, "chime_critical.wav")
    critical_tones = [
        (1046.50, 0.10, 0.85),  # C6
        (0.0, 0.04, 0.0),       # Pause
        (1318.51, 0.12, 0.90),  # E6
        (0.0, 0.04, 0.0),       # Pause
        (1046.50, 0.16, 0.85),  # C6
    ]
    generate_synthesized_wav(critical_path, critical_tones)


if __name__ == "__main__":
    generate_all_sound_assets()
