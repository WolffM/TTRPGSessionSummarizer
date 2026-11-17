# TTRPG Session Summarizer

This is a single Python script, `transcript.py`, not a hosted app: you run it
from the command line against a YouTube URL and your own OpenAI API key, and
it writes two summary files to disk.

Point it at a recording of a tabletop session — it was written for a
Pathfinder group's recorded games. It downloads the audio, transcribes it
with Whisper, and asks GPT for a long-form, chronological recap and a short
TL;DR, each saved as a text file next to the others under `outputs/`.

- [Features](features.md): what each step does and how to run it.
- [Contributing](contributing.md): running it locally, how it's laid out, its one CI check.
- [History](history.md): a rename, an AI-boilerplate README, and a long quiet spell.
- [Trivia](trivia.md)

## Trivia

The repo wasn't always called this: its README and its own git remote once
read `ytTranscriptToWhisper` ([history](history.md#a-rename)).
