# TTRPG Session Summarizer: features

What each step of the pipeline does and how to run it. How it's built is in
[contributing.md](contributing.md).

Everything runs through `summarize_youtube_video()` in `transcript.py`,
called from the bottom of that same file with a hard-coded YouTube URL and
an output directory — there is no command-line argument parsing.

## YouTube download

**What it does.** `youtube_to_mp3()` fetches the audio track of a YouTube
video with `yt-dlp` and converts it to MP3 with `ffmpeg`. If the first
download attempt raises a `DownloadError`, it is retried once automatically.

**How to use it.** Edit the `youtube_url` variable near the bottom of
`transcript.py` to the session recording's URL, then run the script. If a
run has already placed an `.mp3` in the video's output folder, the download
step is skipped and that file is reused.

**What it does not do.** There is no playlist support and no way to pass the
URL on the command line; it is a single variable in the source file.

## Chunked transcription

**What it does.** `chunk_audio()` loads the downloaded audio with `librosa`
and splits it into fixed 10-minute segments, so each piece stays under
Whisper's per-request size limit. `transcribe_audio()` then sends each
segment to OpenAI's Whisper (`whisper-1`) and concatenates the results into
one transcript file. A `RateLimitError` from OpenAI is caught and retried
after a 60-second pause, in a loop with no retry limit.

**How to use it.** No action needed beyond running the script; chunking and
transcription happen automatically between download and summarization. If
`transcripts.txt` already exists for a video, this whole step is skipped.

**What it does not do.** The 10-minute segment length is a constant in the
code, not a setting; changing it means editing `segment_length` in
`summarize_youtube_video()`.

## Two summary formats

**What it does.** `summarize()` sends each transcript chunk to a GPT model
(`gpt-3.5-turbo` by default) with a system prompt asking for a chronological,
narrated, bullet-point recap that calls out key actions, plot developments,
new characters and comedic moments. The chunk summaries are joined into one
long-form summary, then that combined text is summarized a second time with
a stricter "no more than ~30 bullets" prompt to produce a short TL;DR.

**How to use it.** Running the script writes both: `summary_long.txt` (the
long-form recap) and `summary.txt` (the TL;DR), one per video, under
`outputs/<video_id>/`.

**What it does not do.** The two system prompts are Python string literals
in `transcript.py`; there is no config file or CLI flag for tone or focus —
changing what the summary emphasizes means editing the prompt text directly,
as the README's "Customization" section describes.

## Resumable runs

**What it does.** Every stage checks whether its output already exists
before doing the work again: a downloaded `.mp3`, existing audio chunks, an
existing `transcripts.txt`, or an existing `summary.txt` each short-circuit
their step.

**How to use it.** Re-running the script against the same `outputs_dir` for
a video you've already processed is safe and cheap — it reuses whatever
finished last time and only does the remaining work.

**What it does not do.** The resume check on the final TL;DR step
(`summary_file`) is not safe to actually resume from: `summarize_youtube_video()`
only builds the `summaries` list inside the branch that runs when
`summary_file` does *not* yet exist, but then reads that same `summaries`
variable unconditionally afterward to build the TL;DR. Re-running the
script for a video whose `summary.txt` already exists raises a
`NameError: name 'summaries' is not defined` instead of reusing it
(`transcript.py`, `summarize_youtube_video`).

## Trivia

The script still speaks the pre-1.0 OpenAI Python SDK (`openai.Audio.transcribe`,
`openai.ChatCompletion.create`); see
[contributing.md](contributing.md#run-it-locally) for what that means for
running it today.
