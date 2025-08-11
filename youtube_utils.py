import os
import re
import tempfile
from urllib.parse import urlparse, parse_qs
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)
from pydub import AudioSegment
from typing import Optional
from google.cloud import speech, storage

# Google Cloud setup
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")  # Must be set for async STT
LANGUAGES = ["en-US", "hi-IN"]

# Limits
MAX_VIDEO_LENGTH_SEC = 7200  # 2 hours
SYNC_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB for sync STT

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


def _transcribe_google_sync(path: str) -> Optional[str]:
    """Synchronous STT for small files."""
    client = speech.SpeechClient()
    with open(path, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=44100,
        language_code=LANGUAGES[0],
        alternative_language_codes=LANGUAGES[1:],
        enable_automatic_punctuation=True
    )

    print("[Google STT] Sync request sending...")
    response = client.recognize(config=config, audio=audio)
    transcript = " ".join(result.alternatives[0].transcript for result in response.results)
    return transcript if transcript.strip() else None


def _transcribe_google_async(path: str) -> Optional[str]:
    """Async STT for large files using GCS."""
    if not GCP_BUCKET_NAME:
        print("[Google STT] Missing GCP_BUCKET_NAME for async transcription.")
        return None

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCP_BUCKET_NAME)
    blob_name = os.path.basename(path)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path)
    gcs_uri = f"gs://{GCP_BUCKET_NAME}/{blob_name}"
    print(f"[Google STT] Uploaded to {gcs_uri}")

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=44100,
        language_code=LANGUAGES[0],
        alternative_language_codes=LANGUAGES[1:],
        enable_automatic_punctuation=True
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    print("[Google STT] Waiting for operation to complete...")
    response = operation.result(timeout=1800)  # up to 30 min

    transcript = " ".join(result.alternatives[0].transcript for result in response.results)
    blob.delete()  # Clean up
    return transcript if transcript.strip() else None


def get_youtube_transcript(video_id: str) -> Optional[str]:
    """Get transcript via captions or Google STT fallback."""
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

    # Google STT fallback
    try:
        print(f"[Fallback] Google STT transcription for {video_id}")
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        if yt.length > MAX_VIDEO_LENGTH_SEC:
            print(f"[Video Too Long] {yt.length} sec")
            return None

        audio_stream = _choose_audio_stream(yt)
        if not audio_stream:
            print(f"[No Audio Stream] {video_id}")
            return None

        subtype = getattr(audio_stream, "subtype", "") or "mp4"

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = os.path.join(tmp_dir, f"audio.{subtype}")
            audio_stream.download(output_path=tmp_dir, filename=f"audio.{subtype}")

            mp3_path = os.path.join(tmp_dir, "audio.mp3")
            AudioSegment.from_file(raw_path).export(mp3_path, format="mp3")
            file_size = os.path.getsize(mp3_path)
            print(f"[Converted to MP3] size={file_size} bytes")

            if file_size <= SYNC_MAX_FILE_BYTES:
                return _transcribe_google_sync(mp3_path)
            else:
                return _transcribe_google_async(mp3_path)

    except Exception as e:
        print(f"[Google STT Error] {e}")
        return None

    print(f"[Transcript Not Available] {video_id}")
    return None
