# youtube_utils.py
import os
import re
import tempfile
import requests
from urllib.parse import urlparse, parse_qs
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)

# Whisper fallback API key (optional)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_youtube_video_id(url: str) -> str | None:
    """
    Extracts the YouTube video ID from various URL formats.
    """
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0].split("&")[0]
    elif "youtube.com/watch?v=" in url:
        query = urlparse(url).query
        params = parse_qs(query)
        return params.get('v', [None])[0]
    elif "youtube.com/embed/" in url:
        return url.split("youtube.com/embed/")[1].split("?")[0].split("&")[0]
    elif "youtube.com/v/" in url:
        return url.split("youtube.com/v/")[1].split("?")[0].split("&")[0]
    elif "youtube.com/shorts/" in url:
        return url.split("youtube.com/shorts/")[1].split("?")[0].split("&")[0]
    else:
        match = re.search(
            r'(?:v=|/videos/|embed/|youtu\.be/|/v/|watch\?v=|\.be/)([a-zA-Z0-9_-]{11})',
            url
        )
        if match:
            return match.group(1)
    return None


def get_youtube_transcript(video_id: str) -> str | None:
    """
    Retrieves the transcript for a YouTube video.
    Tries manual captions, then auto-generated captions.
    Falls back to Whisper transcription if captions are unavailable (optional).
    """
    # Step 1: Try YouTube captions
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Manual transcript
        try:
            transcript = transcript_list.find_transcript(['en', 'hi'])
        except NoTranscriptFound:
            transcript = None

        # Auto-generated transcript
        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(['en', 'hi'])
            except NoTranscriptFound:
                transcript = None

        if transcript:
            fetched_transcript = transcript.fetch()
            return " ".join(item["text"] for item in fetched_transcript)

    except (TranscriptsDisabled, NoTranscriptFound):
        pass
    except Exception as e:
        print(f"[Transcript Error] {e}")

    # Step 2: Whisper fallback
    if GROQ_API_KEY:
        try:
            print("[Fallback] Using Whisper via Groq API...")
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            audio_stream = yt.streams.filter(only_audio=True).first()

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                audio_stream.download(filename=temp_file.name)
                temp_file_path = temp_file.name

            with open(temp_file_path, "rb") as f:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    files={"file": f},
                    data={"model": "whisper-large-v3"}
                )
            os.remove(temp_file_path)

            if resp.status_code == 200:
                return resp.json().get("text")
            else:
                print(f"[Whisper Error] {resp.status_code} {resp.text}")

        except Exception as e:
            print(f"[Whisper Error] {e}")

    return None
