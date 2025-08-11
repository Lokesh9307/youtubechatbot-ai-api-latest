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
from pydub import AudioSegment
import math
import dotenv
from typing import Optional

dotenv.load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Limits
MAX_VIDEO_LENGTH_SEC = 7200  # 2 hours
GROQ_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB limit for Groq

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
    """Prefer webm audio stream, fallback to lowest bitrate audio."""
    audio_streams = yt.streams.filter(only_audio=True).order_by('abr')
    for s in audio_streams:
        subtype = getattr(s, "subtype", "")
        if "webm" in subtype:
            return s
    return audio_streams.first() if audio_streams else None


def _transcribe_file(path: str, mime: str, filename: str) -> Optional[str]:
    """Send a single audio file to Groq Whisper."""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    data = {"model": "groq/whisper-large-v3"}
    with open(path, "rb") as f:
        files = {"file": (filename, f, mime)}
        resp = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
            timeout=120
        )
    if resp.status_code == 200:
        return resp.json().get("text")
    else:
        print(f"[Whisper API Error] {resp.status_code} — {resp.text}")
        return None


def _chunk_and_transcribe(path: str, mime: str, filename: str) -> Optional[str]:
    """Split audio into <100MB chunks, transcribe each, merge text."""
    audio = AudioSegment.from_file(path)
    size_per_ms = os.path.getsize(path) / len(audio)  # bytes per ms
    max_chunk_ms = int((GROQ_MAX_FILE_BYTES - 1024 * 10) / size_per_ms)  # buffer a little
    chunks = math.ceil(len(audio) / max_chunk_ms)
    print(f"[Audio Chunking] Splitting into {chunks} parts")

    texts = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(chunks):
            start_ms = i * max_chunk_ms
            end_ms = min((i + 1) * max_chunk_ms, len(audio))
            chunk_audio = audio[start_ms:end_ms]
            chunk_filename = f"{filename}_part{i+1}.mp3"
            chunk_path = os.path.join(tmp_dir, chunk_filename)
            chunk_audio.export(chunk_path, format="mp3")

            part_text = _transcribe_file(chunk_path, "audio/mpeg", chunk_filename)
            if part_text:
                texts.append(part_text)
            else:
                print(f"[Chunk Transcription Failed] Part {i+1}")
    return " ".join(texts) if texts else None


def get_youtube_transcript(video_id: str) -> Optional[str]:
    """Get transcript via captions or Whisper fallback."""
    # Try captions first
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
            print(f"[Transcript Found] {video_id} (captions)")
            return " ".join(item['text'] for item in transcript.fetch())
        print(f"[No Captions] {video_id}")
    except (TranscriptsDisabled, NoTranscriptFound):
        print(f"[Captions Disabled] {video_id}")
    except Exception as e:
        print(f"[Transcript Error] {e}")

    # Whisper fallback
    if not GROQ_API_KEY:
        print(f"[No GROQ_API_KEY] Can't use Whisper for {video_id}")
        return None

    try:
        print(f"[Fallback] Whisper transcription for {video_id}")
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        if yt.length > MAX_VIDEO_LENGTH_SEC:
            print(f"[Video Too Long] {yt.length} sec")
            return None

        audio_stream = _choose_audio_stream(yt)
        if not audio_stream:
            print(f"[No Audio Stream] {video_id}")
            return None

        subtype = getattr(audio_stream, "subtype", "") or "mp4"
        mime = "audio/webm" if "webm" in subtype else "audio/mp4"

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = os.path.join(tmp_dir, f"audio.{subtype}")
            audio_stream.download(output_path=tmp_dir, filename=f"audio.{subtype}")

            file_size = os.path.getsize(temp_path)
            print(f"[Downloaded Audio] size={file_size} bytes")

            if file_size <= GROQ_MAX_FILE_BYTES:
                return _transcribe_file(temp_path, mime, f"audio.{subtype}")
            else:
                print(f"[File Too Large] {file_size} bytes — chunking")
                return _chunk_and_transcribe(temp_path, mime, "audio_chunk")

    except Exception as e:
        print(f"[Whisper Error] {e}")
        return None

    print(f"[Transcript Not Available] {video_id}")
    return None
