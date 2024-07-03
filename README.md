# TTRPG Session Summarizer

This project automates the creation of detailed and concise summaries from YouTube recordings of Pathfinder tabletop role-playing game (TTRPG) sessions. By leveraging OpenAI's Whisper for transcription and GPT models for summarization, it streamlines the process of capturing key moments and plot developments from your adventures.

## Features

- **YouTube Integration:** Easily download audio from YouTube videos.
- **Automatic Transcription:** Convert audio into text using Whisper.
- **Intelligent Chunking:** Divide long audio files into manageable segments.
- **Pathfinder-Focused Summaries:** Extract key actions, plot points, character introductions, and comedic highlights.
- **Two Summary Formats:** Generate both a detailed, long-form summary and a concise TL;DR version.
- **Customizable Prompts:** Tailor the AI's summarization style to your preferences.
- **Local Storage:** Save transcripts and summaries for future reference.

## Usage

1. **Set up your .env file:**

   ```
   DISCORD_TOKEN = "YOUR_DISCORD_TOKEN"
   OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
   ```

2. **Run the script:**
   ```bash
   python summarizer.py
   ```
   (Replace `"https://www.youtube.com/watch?v=vPuD-BMn50U"` with the actual YouTube URL of your Pathfinder session.)

3. **Find your summaries:**
   - The detailed summary will be in `outputs/<video_id>/summary_long.txt`.
   - The TL;DR summary will be in `outputs/<video_id>/summary.txt`.

## Customization

- Modify the `system_prompt` variables in the `summarize` function to adjust the focus of the summaries (e.g., emphasize combat encounters, social interactions, exploration, etc.).
- Experiment with different GPT models (e.g., `gpt-4`) for potentially enhanced summarization quality.
- Explore additional output formats or integrations (e.g., generate Markdown summaries, post summaries to Discord, etc.).

## Credits

- This project is based on the "youGPTube" Colab notebook by jerpint: [https://colab.research.google.com/github/jerpint/jerpint.github.io/blob/master/colabs/youGPTube.ipynb#scrollTo=RXymVLFhELLc](https://colab.research.google.com/github/jerpint/jerpint.github.io/blob/master/colabs/youGPTube.ipynb#scrollTo=RXymVLFhELLc)

## Disclaimer

- The quality of summaries may vary depending on the audio quality, clarity of speech, and complexity of the Pathfinder session.
- Please be respectful of copyright when using YouTube content.

## Contributing

Contributions are welcome! Feel free to open issues, submit pull requests, or suggest improvements.
