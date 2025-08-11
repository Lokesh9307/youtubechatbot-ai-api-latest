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
import dotenv

# Load environment variables
dotenv.load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Limits
MAX_VIDEO_LENGTH_SEC = 7200  # 2 hours
MAX_ESTIMATED_AUDIO_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB


def get_youtube_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
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
    Tries: manual captions -> auto-generated -> Whisper (Groq) fallback (if GROQ_API_KEY set).
    Logs every major decision so you can see why something failed.
    Returns the transcript text or None.
    """
    # Try captions first
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_transcript(['en', 'hi'])
        except NoTranscriptFound:
            transcript = None

        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(['en', 'hi'])
            except NoTranscriptFound:
                transcript = None

        if transcript:
            print(f"[Transcript Found] Video ID: {video_id} (captions retrieved)")
            fetched = transcript.fetch()
            return " ".join(item.text for item in fetched)

        print(f"[No Captions] Video ID: {video_id} — No manual or auto-generated captions found.")

    except (TranscriptsDisabled, NoTranscriptFound):
        print(f"[Captions Disabled] Video ID: {video_id} — Subtitles are disabled.")
    except Exception as e:
        print(f"[Transcript Error] {e}")

    # Whisper fallback via Groq
    if not GROQ_API_KEY:
        print(f"[No GROQ_API_KEY] Cannot use Whisper fallback for Video ID: {video_id}")
        print(f"[Transcript Not Available] Video ID: {video_id} — All retrieval methods failed.")
        return None

    try:
        print(f"[Fallback] Using Whisper transcription for Video ID: {video_id}")
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")

        # Length check
        if yt.length > MAX_VIDEO_LENGTH_SEC:
            print(f"[Video Too Long] Video ID: {video_id} — Exceeds max length ({yt.length} sec > {MAX_VIDEO_LENGTH_SEC} sec)")
            return None

        # Select lowest bitrate audio
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').first()
        if audio_stream is None:
            print(f"[No Audio Stream] Video ID: {video_id}")
            return None

        # Estimate size
        abr_kbps = int(audio_stream.abr.replace('kbps', ''))
        estimated_size = (abr_kbps * 1024 / 8) * yt.length
        if estimated_size > MAX_ESTIMATED_AUDIO_SIZE_BYTES:
            print(f"[Audio Too Large] Video ID: {video_id} — Estimated size ({estimated_size / (1024**3):.2f} GB > {MAX_ESTIMATED_AUDIO_SIZE_BYTES / (1024**3):.2f} GB)")
            return None

        # Download audio in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_filename = "audio.mp4"
            temp_path = os.path.join(tmp_dir, temp_filename)

            audio_stream.download(output_path=tmp_dir, filename=temp_filename)

            # Check if file exists and is not empty
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                print(f"[Audio Download Error] No audio file saved for Video ID: {video_id}")
                return None

            # Send to Whisper API
            with open(temp_path, "rb") as f:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    files={"file": (temp_filename, f, "audio/mp4")},
                    data={"model": "whisper-large-v3"}
                )

        if resp.status_code == 200:
            print(f"[Whisper Success] Video ID: {video_id} — Audio transcribed successfully.")
            return resp.json().get("text")
        else:
            print(f"[Whisper API Error] {resp.status_code} — {resp.text}")

    except Exception as e:
        print(f"[Whisper Error] {e}")

    print(f"[Transcript Not Available] Video ID: {video_id} — All retrieval methods failed.")
    return None
