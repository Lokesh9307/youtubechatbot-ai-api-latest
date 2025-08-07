import re
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

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
        match = re.search(r'(?:v=|/videos/|embed/|youtu.be/|\/v\/|watch\?v=|\.be\/)([a-zA-Z0-9_-]{11})', url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id: str) -> str | None:
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['en', 'hi'])
        except Exception:
            transcript = next(iter(transcript_list), None)

        if transcript:
            fetched_transcript = transcript.fetch()
            return " ".join([item.text for item in fetched_transcript])
        else:
            return None
    except Exception as e:
        print(f"[Transcript Error] {e}")
        return None
