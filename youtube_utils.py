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
from typing import Optional

dotenv.load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Limits
MAX_VIDEO_LENGTH_SEC = 7200  # 2 hours (protective)
GROQ_MAX_FILE_BYTES = 100 * 1024 * 1024  # Groq accepts up to 100 MB for direct uploads (see docs).

def get_youtube_video_id(url: str) -> Optional[str]:
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


def _choose_audio_stream(yt: YouTube):
    """
    From pytube YouTube object choose a suitable audio-only stream.
    Prefer webm (widely accepted) then fallback to mp4/m4a.
    Returns stream object or None.
    """
    audio_streams = yt.streams.filter(only_audio=True).order_by('abr')
    # Prefer webm subtype if available
    for s in audio_streams:
        try:
            subtype = getattr(s, "subtype", "")
            mime = getattr(s, "mime_type", "")
        except Exception:
            subtype = ""
            mime = ""
        if "webm" in subtype or "webm" in mime:
            return s
    # Fallback to lowest-bitrate audio if no webm
    try:
        return audio_streams.first()
    except Exception:
        return None


def _stream_filesize_bytes(stream) -> Optional[int]:
    """
    Try to get an accurate filesize from pytube stream (filesize or filesize_approx),
    otherwise return None.
    """
    size = getattr(stream, "filesize", None)
    if size:
        return int(size)
    size_approx = getattr(stream, "filesize_approx", None)
    if size_approx:
        return int(size_approx)
    return None


def get_youtube_transcript(video_id: str) -> Optional[str]:
    """
    Tries: manual captions -> auto-generated -> Whisper (Groq) fallback (if GROQ_API_KEY set).
    Logs every major decision so you can see why something failed.
    Returns the transcript text or None.
    """
    # 1) Try captions via youtube_transcript_api
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list_transcripts(video_id)

        transcript = None
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
            return " ".join(item['text'] for item in fetched)

        print(f"[No Captions] Video ID: {video_id} — No manual or auto-generated captions found.")

    except (TranscriptsDisabled, NoTranscriptFound):
        print(f"[Captions Disabled] Video ID: {video_id} — Subtitles are disabled.")
    except Exception as e:
        print(f"[Transcript Error] {e}")

    # 2) Whisper fallback via Groq
    if not GROQ_API_KEY:
        print(f"[No GROQ_API_KEY] Cannot use Whisper fallback for Video ID: {video_id}")
        print(f"[Transcript Not Available] Video ID: {video_id} — All retrieval methods failed.")
        return None

    try:
        print(f"[Fallback] Using Whisper transcription for Video ID: {video_id}")
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")

        # Basic length check
        if getattr(yt, "length", 0) and yt.length > MAX_VIDEO_LENGTH_SEC:
            print(f"[Video Too Long] Video ID: {video_id} — Exceeds max length ({yt.length} sec > {MAX_VIDEO_LENGTH_SEC} sec)")
            return None

        audio_stream = _choose_audio_stream(yt)
        if audio_stream is None:
            print(f"[No Audio Stream] Video ID: {video_id}")
            return None

        # Try to get filesize from stream metadata if available
        filesize = _stream_filesize_bytes(audio_stream)
        if filesize is not None:
            print(f"[Audio Stream Size] Video ID: {video_id} — stream filesize={filesize} bytes")
            if filesize > GROQ_MAX_FILE_BYTES:
                print(f"[Audio Too Large] Video ID: {video_id} — stream filesize {filesize} bytes > {GROQ_MAX_FILE_BYTES} bytes (Groq limit).")
                return None

        # derive extension and mime type
        subtype = getattr(audio_stream, "subtype", "") or ""
        if "webm" in subtype:
            ext = ".webm"
            mime = "audio/webm"
        elif "mp3" in subtype:
            ext = ".mp3"
            mime = "audio/mpeg"
        elif "mp4" in subtype or "m4a" in subtype:
            # m4a is safer for mp4-audio streams
            ext = ".m4a"
            mime = "audio/mp4"
        else:
            # fallback
            ext = f".{subtype or 'audio'}"
            mime = "application/octet-stream"

        # Download to temporary directory with correct extension
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_filename = "audio" + ext
            temp_path = os.path.join(tmp_dir, temp_filename)

            try:
                audio_stream.download(output_path=tmp_dir, filename=temp_filename)
            except Exception as dl_e:
                # pytube sometimes fails on specific streams; log and abort
                print(f"[Audio Download Error] Video ID: {video_id} — download failed: {dl_e}")
                return None

            # verify file exists and size
            if not os.path.exists(temp_path):
                print(f"[Audio Download Error] Video ID: {video_id} — file not found after download: {temp_path}")
                return None

            actual_size = os.path.getsize(temp_path)
            print(f"[Audio Downloaded] Video ID: {video_id} — path={temp_path} size={actual_size} bytes")

            if actual_size == 0:
                print(f"[Audio Download Error] Video ID: {video_id} — downloaded file is empty")
                return None

            if actual_size > GROQ_MAX_FILE_BYTES:
                print(f"[Audio Too Large After Download] Video ID: {video_id} — downloaded file {actual_size} bytes > {GROQ_MAX_FILE_BYTES} bytes (Groq limit).")
                return None

            # Prepare multipart/form-data for Groq
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            # Use the Groq-compatible model identifier
            data = {"model": "groq/whisper-large-v3"}

            with open(temp_path, "rb") as f:
                files = {"file": (temp_filename, f, mime)}
                try:
                    resp = requests.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=120
                    )
                except requests.RequestException as re:
                    print(f"[Whisper Request Error] Video ID: {video_id} — network error: {re}")
                    return None

            if resp.status_code == 200:
                # successful transcription
                try:
                    text = resp.json().get("text")
                except Exception:
                    text = None
                if text:
                    print(f"[Whisper Success] Video ID: {video_id} — Audio transcribed successfully.")
                    return text
                else:
                    print(f"[Whisper Success but no text] Video ID: {video_id} — resp body: {resp.text}")
                    return None
            else:
                print(f"[Whisper API Error] Video ID: {video_id} — status={resp.status_code} body={resp.text}")
                # 400 Bad Request is likely due to file size, unsupported format, or bad model param
                return None

    except Exception as e:
        print(f"[Whisper Error] {e}")

    print(f"[Transcript Not Available] Video ID: {video_id} — All retrieval methods failed.")
    return None
