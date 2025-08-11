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

# New constants for limits
MAX_VIDEO_LENGTH_SEC = 7200  # 2 hours max; adjust as needed
MAX_ESTIMATED_AUDIO_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB max

def get_youtube_video_id(url: str) -> str | None:
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
    Tries: manual captions -> auto-generated -> Whisper (Groq) fallback (if GROQ_API_KEY set)
    Logs every major decision so you can see why something failed.
    Returns the full transcript text or None.
    """
    # Try captions first (unchanged)
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list_transcripts(video_id)  # Fixed: use list_transcripts()

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
            return " ".join(item['text'] for item in fetched)  # Fixed: use 'text' key

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

        # Add length check early
        if yt.length > MAX_VIDEO_LENGTH_SEC:
            print(f"[Video Too Long] Video ID: {video_id} — Exceeds max length ({yt.length} sec > {MAX_VIDEO_LENGTH_SEC} sec)")
            return None

        # Select lowest bitrate audio to minimize size
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').first()  # Lowest ABR first
        if audio_stream is None:
            print(f"[No Audio Stream] Video ID: {video_id}")
            return None

        # Estimate size (abr in kbps, convert to bytes)
        abr_kbps = int(audio_stream.abr.replace('kbps', ''))
        estimated_size = (abr_kbps * 1024 / 8) * yt.length  # bytes (bitrate / 8 = bytes/sec)
        if estimated_size > MAX_ESTIMATED_AUDIO_SIZE_BYTES:
            print(f"[Audio Too Large] Video ID: {video_id} — Estimated size ({estimated_size / (1024**3):.2f} GB > {MAX_ESTIMATED_AUDIO_SIZE_BYTES / (1024**3):.2f} GB)")
            return None

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            audio_stream.download(output_path=temp_file.name)  # Fixed: use output_path for temp
            temp_path = temp_file.name

        with open(temp_path, "rb") as f:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("audio.mp4", f, "audio/mp4")},
                data={"model": "whisper-large-v3"}
            )

        try:
            os.remove(temp_path)
        except Exception:
            pass

        if resp.status_code == 200:
            print(f"[Whisper Success] Video ID: {video_id} — Audio transcribed successfully.")
            return resp.json().get("text")
        else:
            # Log full response body for debugging (important)
            print(f"[Whisper API Error] {resp.status_code} — {resp.text}")

    except Exception as e:
        print(f"[Whisper Error] {e}")

    print(f"[Transcript Not Available] Video ID: {video_id} — All retrieval methods failed.")
    return None