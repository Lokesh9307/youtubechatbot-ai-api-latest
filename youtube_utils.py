# youtube_utils.py
import os
import re
import tempfile
import time
from typing import Optional
from urllib.parse import urlparse, parse_qs

from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pydub import AudioSegment
import dotenv

# google cloud clients (lazy init below)
from google.cloud import storage
from google.cloud import speech

dotenv.load_dotenv()

# Config
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "transcripts-store")
MAX_VIDEO_LENGTH_SEC = int(os.getenv("MAX_VIDEO_LENGTH_SEC", 7200))  # protective
GCS_AUDIO_PREFIX = os.getenv("GCS_AUDIO_PREFIX", "audio_uploads")     # folder in bucket
GCS_TRANSCRIPT_PREFIX = os.getenv("GCS_TRANSCRIPT_PREFIX", "transcripts")

# Lazy clients (initialized on first use)
_storage_client: Optional[storage.Client] = None
_speech_client: Optional[speech.SpeechClient] = None


def _get_storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def _get_speech_client() -> speech.SpeechClient:
    global _speech_client
    if _speech_client is None:
        _speech_client = speech.SpeechClient()
    return _speech_client


# ------------------------------
# Utilities: YouTube ID + streams
# ------------------------------
def get_youtube_video_id(url: str) -> Optional[str]:
    """Extract YouTube video id from many URL formats."""
    if not url:
        return None
    # quick patterns
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0].split("&")[0]
    if "youtube.com/watch?v=" in url:
        q = urlparse(url).query
        params = parse_qs(q)
        return params.get("v", [None])[0]
    if "youtube.com/embed/" in url:
        return url.split("youtube.com/embed/")[1].split("?")[0].split("&")[0]
    if "youtube.com/v/" in url:
        return url.split("youtube.com/v/")[1].split("?")[0].split("&")[0]
    # fallback regex for 11-char id
    m = re.search(r"(?:v=|/videos/|embed/|youtu\.be/|/v/|watch\?v=|\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def _choose_audio_stream(yt: YouTube):
    """Return best audio-only stream (prefer webm if present)."""
    streams = yt.streams.filter(only_audio=True).order_by("abr")
    for s in streams:
        if "webm" in getattr(s, "subtype", ""):
            return s
    return streams.first() if streams else None


# ------------------------------
# GCS helpers
# ------------------------------
def _gcs_audio_path(video_id: str) -> str:
    """GCS object path (folder + filename) for audio uploads."""
    return f"{GCS_AUDIO_PREFIX}/{video_id}.flac"


def _gcs_transcript_path(video_id: str) -> str:
    """GCS object path for transcripts (.txt)."""
    return f"{GCS_TRANSCRIPT_PREFIX}/{video_id}.txt"


def _upload_file_to_gcs(local_path: str, bucket_name: str, object_path: str) -> str:
    """Upload a file to GCS and return the gs:// URI."""
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_path)
    blob.upload_from_filename(local_path)
    uri = f"gs://{bucket_name}/{object_path}"
    print(f"[GCS] Uploaded {local_path} -> {uri}")
    return uri


def _fetch_transcript_from_gcs(video_id: str) -> Optional[str]:
    """If transcript exists in GCS, return it (cached)."""
    try:
        client = _get_storage_client()
        bucket = client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(_gcs_transcript_path(video_id))
        if blob.exists():
            print(f"[GCS] Found cached transcript for {video_id}")
            return blob.download_as_text()
    except Exception as e:
        print(f"[GCS] Error checking cached transcript: {e}")
    return None


def _save_transcript_to_gcs(video_id: str, transcript: str):
    """Save transcript text to GCS (overwrite)."""
    try:
        client = _get_storage_client()
        bucket = client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(_gcs_transcript_path(video_id))
        blob.upload_from_string(transcript, content_type="text/plain")
        print(f"[GCS] Saved transcript for {video_id}")
    except Exception as e:
        print(f"[GCS] Error saving transcript: {e}")


# ------------------------------
# Transcription helpers
# ------------------------------
def _convert_to_flac_16k_mono(input_path: str, out_path: str) -> bool:
    """
    Convert any audio file to 16kHz mono FLAC (Google-friendly).
    Returns True on success.
    """
    try:
        seg = AudioSegment.from_file(input_path)
        seg = seg.set_channels(1).set_frame_rate(16000)
        seg.export(out_path, format="flac")
        size = os.path.getsize(out_path)
        print(f"[Audio] Converted to FLAC (16k/mono) size={size} bytes")
        return size > 0
    except Exception as e:
        print(f"[Audio conversion error] {e}")
        return False


def _transcribe_from_gcs_uri(gcs_uri: str, language_code: str = "en-US", timeout_sec: int = 3600) -> Optional[str]:
    """
    Use long_running_recognize on the GCS URI. Returns transcript or None.
    """
    try:
        client = _get_speech_client()
        audio = speech.RecognitionAudio(uri=gcs_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=16000,
            language_code=language_code,
            enable_automatic_punctuation=True,
        )
        print(f"[STT] Starting long_running_recognize for {gcs_uri}")
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=timeout_sec)
        parts = []
        for result in response.results:
            parts.append(result.alternatives[0].transcript.strip())
        transcript = " ".join(parts).strip()
        print(f"[STT] Completed transcription, length={len(transcript)} chars")
        return transcript if transcript else None
    except Exception as e:
        print(f"[STT] Error during transcription: {e}")
        return None


# ------------------------------
# Main public function
# ------------------------------
def get_youtube_transcript(video_id: str) -> Optional[str]:
    """
    Return transcript for a YouTube video ID:
      1) check cached transcript in GCS
      2) try YouTube captions (manual/auto)
      3) fallback: download audio -> convert -> upload to GCS -> long_running_recognize -> cache
    Returns transcript string or None.
    """
    if not video_id:
        return None

    # 1) cached transcript
    cached = _fetch_transcript_from_gcs(video_id)
    if cached:
        return cached

    # 2) try captions
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        try:
            transcript = transcript_list.find_transcript(["en", "hi"])
        except NoTranscriptFound:
            transcript = None
        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(["en", "hi"])
            except NoTranscriptFound:
                transcript = None
        if transcript:
            print(f"[Captions] Found for {video_id}")
            fetched = transcript.fetch()
            text = " ".join(item.get("text", "") for item in fetched).strip()
            if text:
                _save_transcript_to_gcs(video_id, text)
                return text
    except (TranscriptsDisabled, NoTranscriptFound):
        print(f"[Captions] Disabled or not found for {video_id}")
    except Exception as e:
        print(f"[Captions] Error checking captions: {e}")

    # 3) fallback via Google STT (GCS upload + long_running_recognize)
    print(f"[Fallback] Using Google STT for {video_id}")
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")

        # guard: video length
        if getattr(yt, "length", 0) and yt.length > MAX_VIDEO_LENGTH_SEC:
            print(f"[Fallback] Video too long ({yt.length}s) -> abort")
            return None

        stream = _choose_audio_stream(yt)
        if stream is None:
            print("[Fallback] No audio stream available")
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_name = "audio_raw"
            raw_path = os.path.join(tmpdir, raw_name)
            # pytube download: output_path must be directory and filename param is used
            stream.download(output_path=tmpdir, filename=raw_name)
            # find the actual file (pytube may add extension)
            # search for any file beginning with raw_name in tmpdir
            found = None
            for fname in os.listdir(tmpdir):
                if fname.startswith(raw_name):
                    found = os.path.join(tmpdir, fname)
                    break
            if not found or not os.path.exists(found):
                print("[Fallback] Download failed or file missing")
                return None

            # convert to FLAC 16k mono
            flac_local = os.path.join(tmpdir, f"{video_id}.flac")
            ok = _convert_to_flac_16k_mono(found, flac_local)
            if not ok:
                print("[Fallback] Conversion to FLAC failed")
                return None

            # upload flac to GCS
            gcs_obj = _gcs_audio_path(video_id)
            gcs_uri = _upload_file_to_gcs(flac_local, GCP_BUCKET_NAME, gcs_obj)

            # transcribe from GCS
            transcript = _transcribe_from_gcs_uri(gcs_uri, language_code="en-US", timeout_sec=1800)
            if transcript:
                _save_transcript_to_gcs(video_id, transcript)
                return transcript
            else:
                print("[Fallback] STT returned empty transcript")

    except Exception as e:
        print(f"[Fallback] Unexpected error: {e}")

    return None
