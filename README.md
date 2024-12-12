# Twitch Stream Analyzer with Gemini 2.0 Flash

This project uses Google's Gemini 2.0 Flash model to analyze Twitch streams in real-time, processing both audio and video inputs.

## Requirements

- Python 3.11 or higher
- A Google API key with access to Gemini 2.0 Flash
- PortAudio (required for PyAudio)
  - Windows: Install with `pip install pyaudio`
  - Linux: Install with `sudo apt-get install python3-pyaudio portaudio19-dev`
  - macOS: Install with `brew install portaudio`

## Setup

1. Install Python 3.11:
   ```bash
   # Download and install from python.org or your system's package manager
   ```

2. Install Poetry if you haven't already:
   ```bash
   pip install poetry
   ```

3. Clone this repository and install dependencies:
   ```bash
   poetry install
   ```

4. Create a `.env` file in the project root and add your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

Run the analyzer with a Twitch stream URL:
```bash
poetry run python twitch_gemini_analyzer/twitch_stream_analyzer.py "https://twitch.tv/channel_name"
```

Optional parameters:
- `--quality`: Stream quality (e.g., "best", "720p60", "480p"). Default: "best"

## Features

- Real-time Twitch stream analysis using Gemini 2.0 Flash
- Audio and video stream processing
- Command-line interface for easy usage
- Configurable stream quality

## Troubleshooting

### PyAudio Installation Issues
- If you encounter issues installing PyAudio:
  1. Make sure you have PortAudio installed for your system
  2. On Windows, you might need to install Visual C++ Build Tools
  3. Try installing the wheel directly: `pip install pipwin && pipwin install pyaudio`

### Stream Issues
- If the stream doesn't load:
  1. Check if the Twitch channel is online
  2. Try a different quality setting
  3. Verify your internet connection

### API Issues
- If you get API errors:
  1. Verify your API key in the `.env` file
  2. Check if you have access to Gemini 2.0 Flash
  3. Check your API quota and limits

## Development Status

This project is currently in development. Some features may not be fully implemented or might have issues. Please report any problems in the issues section.
