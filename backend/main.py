import os
import re

import numpy as np
import requests
import sounddevice as sd

from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from datetime import datetime

SAMPLE_RATE = 16000
RECORDINGS_DIR = "../recordings"
NOTES_DIR = "../notes"
OLLAMA_MODEL = "llama3.2"

os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)


def record_audio() -> str:
    filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    path = os.path.join(RECORDINGS_DIR, filename)

    input("Press ENTER to start recording...")
    print("Recording started. Press ENTER again to stop.")

    recording = []
    is_recording = True

    def callback(indata, frames, time, status):
        if status:
            print(status)

        if is_recording:
            recording.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=callback,
    )

    stream.start()
    input()

    is_recording = False
    stream.stop()
    stream.close()

    if not recording:
        raise RuntimeError("No audio was recorded.")

    audio = np.concatenate(recording, axis=0)
    write(path, SAMPLE_RATE, audio)

    print(f"Saved recording: {path}")
    return path


def transcribe_audio(path: str) -> str:
    print("Loading Whisper model...")
    model = WhisperModel("small", device="cpu", compute_type="int8")

    print("Transcribing...")
    segments, info = model.transcribe(path, language="nl")

    transcript = ""
    for segment in segments:
        transcript += segment.text.strip() + " "

    print("Transcript done.")
    return transcript.strip()


def ask_ollama(prompt: str, timeout: int = 120) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=timeout,
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def summarize(transcript: str) -> str:
    prompt = f"""
Je bent MaxEcho, een lokale AI-notulist.

Je taak:
Maak duidelijke gespreksnotities op basis van het transcript.

Belangrijke regels:
- Verzin niets.
- Als iets onduidelijk is, zeg: "Niet duidelijk uit transcript".
- Maak actiepunten concreet.
- Gebruik alleen informatie uit het transcript.
- Schrijf in het Nederlands.
- Gebruik geen placeholders zoals [details].

Geef output in dit format:

## Korte samenvatting
...

## Besproken punten
- ...

## Actiepunten
- [ ] ...

## Open vragen
- ...

Transcript:
{transcript}
"""

    print("Generating local summary with Ollama...")
    return ask_ollama(prompt, timeout=120)


def clean_filename(text: str) -> str:
    text = text.strip().lower()
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "maxecho_notes"


def generate_title(transcript: str) -> str:
    prompt = f"""
Genereer één korte bestandsnaam voor dit gesprek.

Regels:
- maximaal 5 woorden
- lowercase
- gebruik underscores
- geen speciale tekens
- geen extensie toevoegen
- alleen het onderwerp van gesprek
- geef alleen de bestandsnaam terug, geen uitleg

Voorbeeld:
project_planning_meeting

Transcript:
{transcript}
"""

    print("Generating filename...")
    title = ask_ollama(prompt, timeout=60)
    return clean_filename(title)


def save_notes(transcript: str, summary: str) -> str:
    title = generate_title(transcript)
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_prefix}_{title}.md"
    path = os.path.join(NOTES_DIR, filename)

    content = f"""# MaxEcho Notes

Datum: {datetime.now().strftime('%d-%m-%Y %H:%M')}

Bestand: `{filename}`

---

## Transcript

{transcript}

---

{summary}
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Notes saved: {path}")
    return path


if __name__ == "__main__":
    audio_path = record_audio()

    transcript = transcribe_audio(audio_path)

    print("\n--- TRANSCRIPT ---")
    print(transcript)

    if not transcript:
        print("No transcript found. Try speaking louder or checking your microphone.")
        exit()

    summary = summarize(transcript)

    print("\n--- SUMMARY ---")
    print(summary)

    save_notes(transcript, summary)