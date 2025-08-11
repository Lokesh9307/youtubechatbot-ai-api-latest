import os
import re
import tempfile
from urllib.parse import urlparse, parse_qs
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
import dotenv

dotenv.load_dotenv()

# Config
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "transcripts-store")
MAX_VIDEO_LENGTH_SEC = 7200  # 2 hours

def get_youtube_video_id(url: str):
    """Extract video ID from various YouTube URL formats."""
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
    """Choose best available audio stream."""
    audio_streams = yt.streams.filter(only_audio=True).order_by('abr')
    for s in audio_streams:
        if "webm" in getattr(s, "subtype", ""):
            return s
    return audio_streams.first() if audio_streams else None

def _upload_to_gcs(local_path: str, gcs_path: str):
    """Upload local file to GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"[GCS] Uploaded to gs://{BUCKET_NAME}/{gcs_path}")
    return f"gs://{BUCKET_NAME}/{gcs_path}"

def _google_stt_transcribe_gcs(gcs_uri: str) -> str:
    """Transcribe audio from GCS using Google STT."""
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    print("[Google STT] Waiting for operation to complete...")
    response = operation.result(timeout=3600)

    full_text = []
    for result in response.results:
        full_text.append(result.alternatives[0].transcript)
    return " ".join(full_text).strip()

def get_youtube_transcript(video_id: str):
    """Get transcript via captions or Google STT fallback."""
    # Try captions
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        try:
            transcript = transcript_list.find_transcript(['en', 'hi'])
        except NoTranscriptFound:
            pass
        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(['en', 'hi'])
            except NoTranscriptFound:
                pass
        if transcript:
            print(f"[Transcript Found] {video_id}")
            return " ".join(item['text'] for item in transcript.fetch())
        print(f"[No Captions] {video_id}")
    except (TranscriptsDisabled, NoTranscriptFound):
        print(f"[Captions Disabled] {video_id}")
    except Exception as e:
        print(f"[Transcript Error] {e}")

    # Fallback: Google STT
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
            audio_path = os.path.join(tmp_dir, "audio.webm")
            audio_stream.download(output_path=tmp_dir, filename="audio.webm")

            # Convert to FLAC 16kHz mono for Google STT
            flac_path = os.path.join(tmp_dir, "audio.flac")
            audio_seg = AudioSegment.from_file(audio_path)
            audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)
            audio_seg.export(flac_path, format="flac")

            # Upload to GCS and transcribe
            gcs_uri = _upload_to_gcs(flac_path, f"{video_id}.flac")
            return _google_stt_transcribe_gcs(gcs_uri)

    except Exception as e:
        print(f"[Google STT Error] {e}")
        return None

    return None
