import os
import re
import tempfile
import math
from urllib.parse import urlparse, parse_qs
from typing import Optional

from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from google.cloud import speech, storage
from pydub import AudioSegment
import dotenv

dotenv.load_dotenv()

# Google Cloud setup
GCP_BUCKET = os.getenv("GCP_BUCKET", "transcripts-store")
GCP_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Limits
MAX_VIDEO_LENGTH_SEC = 7200  # 2 hours

# Initialize clients
speech_client = speech.SpeechClient()
storage_client = storage.Client()

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


def _upload_to_gcs(local_path: str, gcs_uri: str):
    """Upload local file to Google Cloud Storage."""
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"[Uploaded to GCS] {gcs_uri}")


def _transcribe_with_google_stt(gcs_uri: str) -> Optional[str]:
    """Transcribe audio from GCS URI using Google Speech-to-Text."""
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True
    )
    operation = speech_client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=600)

    texts = []
    for result in response.results:
        texts.append(result.alternatives[0].transcript)

    return " ".join(texts) if texts else None


def get_youtube_transcript(video_id: str) -> Optional[str]:
    """Get transcript via captions or Google Speech-to-Text fallback."""
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

    # Google Speech-to-Text fallback
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = os.path.join(tmp_dir, "raw_audio")
            audio_stream.download(output_path=tmp_dir, filename="raw_audio")

            # Convert to LINEAR16 PCM WAV (mono, 16kHz)
            wav_path = os.path.join(tmp_dir, "audio.wav")
            audio = AudioSegment.from_file(raw_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(wav_path, format="wav")

            if os.path.getsize(wav_path) == 0:
                print("[Error] Extracted WAV file is empty")
                return None

            # Upload to GCS
            gcs_uri = f"gs://{GCP_BUCKET}/{video_id}_audio.wav"
            _upload_to_gcs(wav_path, gcs_uri)

            # Transcribe
            return _transcribe_with_google_stt(gcs_uri)

    except Exception as e:
        print(f"[Google STT Error] {e}")
        return None

    return None
