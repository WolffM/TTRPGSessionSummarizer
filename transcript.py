import os
import shutil
import time

import librosa
import openai
import soundfile as sf
import youtube_dl
from yt_dlp import YoutubeDL as youtube_dl, DownloadError

def find_audio_files(path, extension=".mp3"):
    """Recursively find all files with extension in path."""
    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(extension):
                audio_files.append(os.path.join(root, f))

    return audio_files

def youtube_to_mp3(youtube_url: str, output_dir: str) -> str:
    """Download the audio from a youtube video, save it to output_dir as an .mp3 file.

    Returns the filename of the savied video.
    """

    # config
    ydl_config = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "verbose": True,
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Downloading video from {youtube_url}")

    try:
        with youtube_dl(ydl_config) as ydl:
            ydl.download([youtube_url])
    except DownloadError:
        # weird bug where youtube-dl fails on the first download, but then works on second try... hacky ugly way around it.
        with youtube_dl(ydl_config) as ydl:
            ydl.download([youtube_url])

    audio_filename = find_audio_files(output_dir)[0]
    return audio_filename

def chunk_audio(filename, segment_length: int, output_dir):
    """segment lenght is in seconds"""

    print(f"Chunking audio to {segment_length} second segments...")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # load audio file
    audio, sr = librosa.load(filename, sr=44100)

    # calculate duration in seconds
    duration = librosa.get_duration(y=audio, sr=sr)

    # calculate number of segments
    num_segments = int(duration / segment_length) + 1

    print(f"Chunking {num_segments} chunks...")

    # iterate through segments and save them
    for i in range(num_segments):
        start = i * segment_length * sr
        end = (i + 1) * segment_length * sr
        segment = audio[start:end]
        sf.write(os.path.join(output_dir, f"segment_{i}.mp3"), segment, sr)

    chunked_audio_files = find_audio_files(output_dir)
    return sorted(chunked_audio_files)

def transcribe_audio(audio_files: list, output_file=None, model="whisper-1") -> list:
    print("converting audio to text...")

    transcripts = []
    for audio_file in audio_files:
        audio = open(audio_file, "rb")
        while True:  # Retry loop
            try:
                response = openai.Audio.transcribe(model, audio)
                transcripts.append(response["text"])
                break  # Exit the loop on success
            except openai.error.RateLimitError:
                print("Rate limit error. Waiting 60 seconds...")
                time.sleep(60)  # Pause for a minute


    if output_file is not None:
        # save all transcripts to a .txt file
        with open(output_file, "w") as file:
            for transcript in transcripts:
                file.write(transcript + "\n")

    return transcripts

def summarize(
    chunks: list[str], system_prompt: str, model="gpt-3.5-turbo", output_file=None
):

    print(f"Summarizing with {model=}")

    summaries = []
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ],
        )
        summary = response["choices"][0]["message"]["content"]
        summaries.append(summary)

    if output_file is not None:
        # save all transcripts to a .txt file
        with open(output_file, "w") as file:
            for summary in summaries:
                file.write(summary + "\n")

    return summaries

def summarize_youtube_video(youtube_url, outputs_dir):
    raw_audio_dir = f"{outputs_dir}/raw_audio/"
    chunks_dir = f"{outputs_dir}/chunks"
    transcripts_file = f"{outputs_dir}/transcripts.txt"
    summary_file = f"{outputs_dir}/summary.txt"
    segment_length = 10 * 60  # chunk to 10 minute segments

    if os.path.exists(outputs_dir):
        # delete the outputs_dir folder and start from scratch
        shutil.rmtree(outputs_dir)
        os.mkdir(outputs_dir)

    # download the video using youtube-dl
    audio_filename = youtube_to_mp3(youtube_url, output_dir=raw_audio_dir)

    # chunk each audio file to shorter audio files (not necessary for shorter videos...)
    chunked_audio_files = chunk_audio(
        audio_filename, segment_length=segment_length, output_dir=chunks_dir
    )

    # transcribe each chunked audio file using whisper speech2text
    transcriptions = transcribe_audio(chunked_audio_files, transcripts_file)

    # summarize each transcription using chatGPT
    system_prompt = """
        You are an expert Pathfinder summarizer. Analyze this transcribed audio chunk from my recorded Pathfinder session and provide a clear, bullet-point summary focusing on:
        Seperate your bullets into sections with meaningful section heads. Don't write more than ~30 bullet points in total. While writing keep a narrative tone, try not to sound overly serious or robotic. Use a prose as if you might be a storyteller.
        Key Actions: What critical decisions did the party make? Did they fight powerful foes, overcome a puzzle, or negotiate a tense situation?
        Plot Developments: Did any major story revelations occur? Did they find crucial clues, uncover a villain's plot, or make progress toward their ultimate goal?
        New Characters: Were any important NPCs introduced? Summarize their name, role, and any key interactions with the party.
        Comedic Moments: Did any hilarious banter, epic fails, or unexpected antics happen? Capture those moments!
        Example to illustrate: If the transcribed chunk includes fighting a big monster and discovering a hidden room, the ideal summary should look something like this:
        In the Caves
        * The party vanquished a great monster in a harrowing battle.
        * In the heat of the moment, Slick broke from the group to lockpick a hidden door, what a sneaky rat!
        * After the battle they made contact with Duncan, a dwarven miner, who told them about the upcoming great danger in the caves.
    """
    summaries = summarize(
        transcriptions, system_prompt=system_prompt, output_file=summary_file
    )

    system_prompt_tldr = """
        You are an expert Pathfinder summarizer. Analyze this transcribed audio chunk from my recorded Pathfinder session and provide a clear, bullet-point summary focusing on:
        Seperate your bullets into sections with meaningful section heads. Don't write more than ~30 bullet points in total. While writing keep a narrative tone, try not to sound overly serious or robotic. Use a prose as if you might be a storyteller.
        Key Actions: What critical decisions did the party make? Did they fight powerful foes, overcome a puzzle, or negotiate a tense situation?
        Plot Developments: Did any major story revelations occur? Did they find crucial clues, uncover a villain's plot, or make progress toward their ultimate goal?
        New Characters: Were any important NPCs introduced? Summarize their name, role, and any key interactions with the party.
        Comedic Moments: Did any hilarious banter, epic fails, or unexpected antics happen? Capture those moments!
        Example to illustrate: If the transcribed chunk includes fighting a big monster and discovering a hidden room, the ideal summary should look something like this:
        In the Caves
        * The party vanquished a great monster in a harrowing battle.
        * In the heat of the moment, Slick broke from the group to lockpick a hidden door, what a sneaky rat!
        * After the battle they made contact with Duncan, a dwarven miner, who told them about the upcoming great danger in the caves.
    """
    # put the entire summary to a single entry
    long_summary = "\n".join(summaries)
    short_summary = summarize(
        [long_summary], system_prompt=system_prompt_tldr, output_file=summary_file
    )[0]

    return long_summary, short_summary

youtube_url = "<PUT YOUTUBE VIDEO LINK>"
outputs_dir = "outputs/"
openai.api_key = '<PUT OPENAI KEY HERE>'

long_summary, short_summary = summarize_youtube_video(youtube_url, outputs_dir)

print("Summaries:")
print("=" * 80)
print("Long summary:")
print("=" * 80)
print(long_summary)
print()

print("=" * 80)
print("Video - TL;DR")
print("=" * 80)
print(short_summary)