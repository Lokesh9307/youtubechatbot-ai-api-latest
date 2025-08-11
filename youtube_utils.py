import os
import re
import tempfile
import math
from typing import Optional
from urllib.parse import urlparse, parse_qs

from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)
from pydub import AudioSegment
from google.cloud import speech

# Limits
MAX_VIDEO_LENGTH_SEC = 7200  # 2 hours

def get_youtube_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various YouTube URL formats."""
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


def _transcribe_with_google_stt(audio_path: str) -> Optional[str]:
    """Transcribe an audio file using Google Cloud Speech-to-Text."""
    client = speech.SpeechClient()

    audio = AudioSegment.from_file(audio_path)
    duration_sec = len(audio) / 1000

    # Convert to FLAC for upload
    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp_flac:
        audio.export(tmp_flac.name, format="flac")
        flac_path = tmp_flac.name

    with open(flac_path, "rb") as f:
        content = f.read()

    gcs_audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=audio.frame_rate,
        language_code="en-US",
        enable_automatic_punctuation=True
    )

    try:
        if duration_sec > 60:
            print(f"[Google STT] Long-running recognition, {duration_sec:.1f}s audio")
            operation = client.long_running_recognize(config=config, audio=gcs_audio)
            response = operation.result(timeout=duration_sec * 2)
        else:
            print(f"[Google STT] Synchronous recognition, {duration_sec:.1f}s audio")
            response = client.recognize(config=config, audio=gcs_audio)

        transcript_parts = []
        for result in response.results:
            transcript_parts.append(result.alternatives[0].transcript.strip())

        os.remove(flac_path)
        return " ".join(transcript_parts) if transcript_parts else None
    except Exception as e:
        print(f"[Google STT Error] {e}")
        return None


def get_youtube_transcript(video_id: str) -> Optional[str]:
    """Get transcript via captions or Google STT fallback."""
    # Try captions first
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

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
    print(f"[Fallback] Google STT transcription for {video_id}")
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        if yt.length > MAX_VIDEO_LENGTH_SEC:
            print(f"[Video Too Long] {yt.length} sec")
            return None

        audio_stream = _choose_audio_stream(yt)
        if not audio_stream:
            print(f"[No Audio Stream] {video_id}")
            return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtype = getattr(audio_stream, "subtype", "") or "mp4"
            temp_path = os.path.join(tmp_dir, f"audio.{subtype}")
            audio_stream.download(output_path=tmp_dir, filename=f"audio.{subtype}")

            return _transcribe_with_google_stt(temp_path)
    except Exception as e:
        print(f"[Google STT Error] {e}")
        return None
