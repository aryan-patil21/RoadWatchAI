"""
RoadWatch AI - Open-Source Audio Asset Generator
------------------------------------------------
Generates platform-agnostic, 100% royalty-free, open-source 16-bit PCM .WAV audio cues.
Now configured with sharp, urgent, high-visibility automotive warning beeps:
- Caution: Rapid dual-pulse acoustic warning beep (880 Hz).
- Critical: Urgent, piercing triple-pulse emergency hazard alarm (1175 Hz - 1480 Hz) at high amplitude.
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
    Synthesizes crisp, punchy square/sine hybrid audio tones with minimal onset delay.
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
            # Sharp automotive alert envelope: rapid 2ms rise, sustained body, crisp release
            rise = min(1.0, i / (sample_rate * 0.005))
            fall = min(1.0, (num_samples - i) / (sample_rate * 0.008))
            envelope = rise * fall

            # Rich harmonic tone (fundamental + 3rd harmonic) for cutting through road noise
            fundamental = math.sin(2.0 * math.pi * freq * t)
            harmonic = 0.25 * math.sin(2.0 * math.pi * (freq * 2.0) * t)
            sample_val = amp * envelope * (0.8 * fundamental + harmonic)
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

    print(f"✓ Generated strong warning sound: {output_path} ({len(total_samples) / sample_rate:.2f}s)")


def generate_all_sound_assets(target_dir: str = "assets/sounds"):
    """Generates sharp, loud Caution beep and Critical alarm cues."""
    # 1. Caution Beep: Sharp two-pulse warning beep (880 Hz A5)
    caution_path = os.path.join(target_dir, "chime_caution.wav")
    caution_tones = [
        (880.0, 0.09, 0.90),   # Pulse 1 (Loud 880 Hz beep)
        (0.0, 0.05, 0.0),      # Short silence
        (880.0, 0.12, 0.95),   # Pulse 2
    ]
    generate_synthesized_wav(caution_path, caution_tones)

    # 2. Critical Alarm: Piercing, rapid triple-beep hazard siren (1200 Hz - 1480 Hz)
    critical_path = os.path.join(target_dir, "chime_critical.wav")
    critical_tones = [
        (1318.5, 0.08, 0.98),  # E6 Beep 1
        (0.0, 0.035, 0.0),     # Pause
        (1479.9, 0.08, 0.98),  # F#6 Beep 2
        (0.0, 0.035, 0.0),     # Pause
        (1318.5, 0.12, 0.98),  # E6 Beep 3
        (0.0, 0.04, 0.0),      # Pause
        (1567.9, 0.14, 1.00),  # G6 Beep 4 (Climax)
    ]
    generate_synthesized_wav(critical_path, critical_tones)


if __name__ == "__main__":
    generate_all_sound_assets()
