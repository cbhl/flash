#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import base64
import io
import os
import sys
import time
import traceback
import urllib.parse
from typing import Optional

import cv2
import ffmpeg
import numpy as np
import PIL.Image
import sounddevice as sd
import streamlink
import typer
from dotenv import load_dotenv
from google import genai

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# Load environment variables
load_dotenv()

# Audio settings
FORMAT = np.int16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
CHUNK_SIZE = 512

# Video settings
FPS = 30.0  # Target 30 fps
FRAME_TIME = 1.0 / FPS
FRAME_BATCH_SIZE = 3  # Send 3 frames at once (represents 100ms of video at 30fps)

MODEL = "models/gemini-2.0-flash-exp"
client = genai.Client(
    http_options={'api_version': 'v1alpha'}
)

CONFIG = {
    "generation_config": {
        "temperature": 0.4,
        "top_p": 0.8,
        "top_k": 40,
        "response_modalities": ["TEXT"]
    }
}

# SYSTEM_PROMPT = """You are watching a Twitch stream. Your task is to:
# 1. Transcribe spoken words exactly as heard
# 2. When asked "describe the scene", provide:
#    a) A brief description in [square brackets]
#    b) A clip assessment in JSON format like:
#       {"is_clip_worthy": true/false, "title": "Suggested Title", "game": "Game Name"}

# Rules:
# - First identify the game being played, then maintain that context
# - Focus on meaningful events and reactions
# - No greetings or meta-commentary
# - Keep descriptions brief and focused on changes
# - For clip assessment:
#   * Clips are worthy if they contain: high emotion, interesting plays, funny moments
#   * Titles should be attention-grabbing but accurate
#   * Always identify the game being played"""

SYSTEM_PROMPT = """Ignore all previous instructions.
You are watching a Twitch stream of enluna ("Annie") playing Teamfight Tactics.

Your task is to:
1. Continuously transcribe spoken words, and then translate them into French.

Format each line of your response as follows:
<original text> // <french translation>

Rules:
- No greetings or meta-commentary
- Make sure to translate each sentence that you transcribe.
"""

class ConsoleManager:
    def __init__(self):
        self.status_line = ""
        
    def print_response(self, text):
        """Print response text above the status line."""
        # Clear the current status line
        print('\r' + ' ' * len(self.status_line), end='\r')
        
        # Print the response text with a newline
        print(text)
        
        # Reprint the status line
        print(self.status_line, end='\r', flush=True)
        
    def update_status(self, status):
        """Update the status line at the bottom."""
        self.status_line = status
        print('\r' + self.status_line, end='', flush=True)

class TwitchStreamAnalyzer:
    def __init__(self, url: str):
        self.url = url
        self.video_out_queue = asyncio.Queue()
        self.text_out_queue = asyncio.Queue()
        self.stream = None
        self.cap = None
        self.session = None
        self.video_packets_sent = 0
        self.audio_packets_sent = 0
        self.last_status_time = 0
        self.console = ConsoleManager()

    def _validate_twitch_url(self) -> bool:
        """Validate if the URL is a valid Twitch URL."""
        try:
            parsed = urllib.parse.urlparse(self.url)
            return parsed.netloc in ["twitch.tv", "www.twitch.tv"]
        except:
            return False

    async def process_frame(self, frame):
        """Process a video frame for the Gemini model."""
        try:
            # Convert frame to PIL Image
            img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.thumbnail([1024, 1024])  # Resize to max dimensions

            # Convert to bytes
            image_io = io.BytesIO()
            img.save(image_io, format="jpeg")
            image_io.seek(0)

            return {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_io.read()).decode()
            }
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    async def print_status(self):
        """Print status information periodically."""
        while True:
            status = f"Video packets: {self.video_packets_sent} | Audio packets: {self.audio_packets_sent}"
            self.console.update_status(status)
            await asyncio.sleep(1)

    async def get_frames(self):
        """Get frames from the Twitch stream."""
        try:
            streams = streamlink.streams(self.url)
            if not streams:
                print("No streams found")
                return

            stream_url = streams["best"].url
            self.cap = await asyncio.to_thread(cv2.VideoCapture, stream_url)
            
            last_frame_time = time.time()

            while True:
                current_time = time.time()
                if current_time - last_frame_time >= FRAME_TIME:
                    ret, frame = await asyncio.to_thread(self.cap.read)
                    if not ret:
                        break

                    frame_data = await self.process_frame(frame)
                    if frame_data:
                        await self.video_out_queue.put((current_time, frame_data))
                        last_frame_time = current_time
                
                await asyncio.sleep(FRAME_TIME / 2)  # Check twice per frame interval

        except Exception as e:
            print(f"Error getting frames: {e}")
            traceback.print_exc()
        finally:
            if self.cap:
                self.cap.release()

    async def send_frames(self):
        """Send frames to the Gemini model."""
        frames_sent = 0
        frame_buffer = []
        last_send_time = time.time()
        
        while True:
            try:
                # Get frame with timestamp
                timestamp, frame = await self.video_out_queue.get()
                frame_buffer.append(frame)
                
                current_time = time.time()
                # Send batch if we have enough frames or enough time has passed
                if len(frame_buffer) >= FRAME_BATCH_SIZE or (current_time - last_send_time >= 0.1):  # 100ms max delay
                    if frame_buffer:
                        self.video_packets_sent += len(frame_buffer)
                        for f in frame_buffer:
                            await self.session.send(f)
                        
                        frames_sent += len(frame_buffer)
                        frame_buffer = []
                        last_send_time = current_time
            
            except Exception as e:
                print(f"Error sending frame: {e}")
                if frame_buffer:
                    frame_buffer = []  # Clear buffer on error

    async def get_audio(self):
        """Get audio from the Twitch stream."""
        try:
            streams = streamlink.streams(self.url)
            if not streams:
                print("No streams found")
                return

            stream_url = streams["audio_only"].url if "audio_only" in streams else streams["best"].url
            
            # Set up ffmpeg process for audio extraction
            process = (
                ffmpeg
                .input(stream_url)
                .output(
                    'pipe:',
                    format='s16le',  # 16-bit PCM
                    acodec='pcm_s16le',
                    ac=1,  # mono
                    ar=SEND_SAMPLE_RATE,  # 16kHz
                    loglevel='error'
                )
                .run_async(pipe_stdout=True)
            )

            # Read audio data in chunks
            while True:
                try:
                    # Read a chunk of audio data
                    audio_chunk = await asyncio.to_thread(
                        process.stdout.read,
                        CHUNK_SIZE * 2  # multiply by 2 because we're reading 16-bit samples
                    )
                    
                    if not audio_chunk:
                        break

                    # Convert to base64
                    audio_b64 = base64.b64encode(audio_chunk).decode()
                    
                    # Create audio message
                    audio_message = {
                        "mime_type": "audio/pcm",
                        "data": audio_b64
                    }
                    
                    await self.text_out_queue.put(audio_message)
                    self.audio_packets_sent += 1
                    
                    # Rate limit to avoid overwhelming the queue
                    # await asyncio.sleep(CHUNK_SIZE / SEND_SAMPLE_RATE)

                except Exception as e:
                    print(f"Error reading audio: {e}")
                    break

        except Exception as e:
            print(f"Error in audio capture: {e}")
        finally:
            if 'process' in locals():
                process.kill()

    async def send_audio(self):
        """Send audio to the Gemini model."""
        audio_chunks_sent = 0
        audio_buffer = []
        while True:
            audio = await self.text_out_queue.get()
            try:
                audio_buffer.append(audio)
                
                # Send batches of 10 audio chunks together
                if len(audio_buffer) >= 10:
                    self.audio_packets_sent += len(audio_buffer)
                    combined_audio = {
                        "mime_type": "audio/pcm",
                        "data": base64.b64encode(b''.join([base64.b64decode(chunk["data"]) for chunk in audio_buffer])).decode()
                    }
                    await self.session.send(combined_audio)
                    audio_chunks_sent += len(audio_buffer)
                    audio_buffer = []
            except Exception as e:
                print(f"Error sending audio: {e}")

    async def prompt_scene_description(self):
        """Periodically ask for scene descriptions."""
        pass
        # while True:
        #     try:
        #         await asyncio.sleep(15.0)  # Every 15 seconds
        #         await self.session.send("describe the scene and assess if this would make a good clip", end_of_turn=True)
        #     except Exception as e:
        #         print(f"Error prompting scene description: {e}")
        #         await asyncio.sleep(1)

    async def receive_responses(self):
        """Receive and process responses from the model."""
        while True:  # Add outer loop to keep receiving
            try:
                async for response in self.session.receive():
                    server_content = response.server_content
                    
                    if server_content is not None and server_content.model_turn is not None:
                        for part in server_content.model_turn.parts:
                            if part.text is not None and part.text.strip():
                                self.console.print_response(part.text)
                    elif response.text and response.text.strip():
                        self.console.print_response(response.text)

            except Exception as e:
                print(f"Error in receive_responses: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)  # Wait before retrying

    async def run(self):
        """Main run loop."""
        if not self._validate_twitch_url():
            print("Invalid Twitch URL")
            return

        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                
                # Send system prompt
                await self.session.send(SYSTEM_PROMPT, end_of_turn=True)
                
                async with asyncio.TaskGroup() as tg:
                    # Status task
                    tg.create_task(self.print_status())
                    
                    # Video tasks
                    tg.create_task(self.get_frames())
                    tg.create_task(self.send_frames())
                    
                    # Audio tasks
                    tg.create_task(self.get_audio())
                    tg.create_task(self.send_audio())
                    
                    # Scene description prompting
                    tg.create_task(self.prompt_scene_description())
                    
                    # Response handling
                    tg.create_task(self.receive_responses())

                    def check_error(task):
                        if task.cancelled():
                            return
                        if task.exception() is not None:
                            e = task.exception()
                            traceback.print_exception(None, e, e.__traceback__)
                            sys.exit(1)

                    for task in tg._tasks:
                        task.add_done_callback(check_error)

        except Exception as e:
            print(f"Error in run: {e}")
            traceback.print_exc()

def main(
    url: str = typer.Argument(..., help="Twitch stream URL"),
):
    """
    Analyze a Twitch stream using Google's Gemini model.
    """
    analyzer = TwitchStreamAnalyzer(url)
    asyncio.run(analyzer.run())

if __name__ == "__main__":
    typer.run(main)
