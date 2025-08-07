from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import os
import re
from urllib.parse import urlparse, parse_qs

app = FastAPI()

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

class VideoRequest(BaseModel):
    url: str

def get_video_id(url: str) -> str | None:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "youtube.com/watch?v=" in url:
        query = urlparse(url).query
        params = parse_qs(query)
        return params.get('v', [None])[0]
    elif "youtube.com/embed/" in url:
        return url.split("youtube.com/embed/")[1].split("?")[0]
    elif "youtube.com/v/" in url:
        return url.split("youtube.com/v/")[1].split("?")[0]
    elif "youtube.com/shorts/" in url:
        return url.split("youtube.com/shorts/")[1].split("?")[0]
    else:
        match = re.search(r'(?:v=|\/)([a-zA-Z0-9_-]{11})', url)
        return match.group(1) if match else None

def get_transcript(video_id: str) -> str | None:
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['en', 'hi'])
        except Exception:
            transcript = next(iter(transcript_list), None)

        if transcript:
            fetched = transcript.fetch()
            return " ".join([item.text for item in fetched])
        return None
    except Exception as e:
        print(f"[Transcript Error] {e}")
        return None

@app.post("/summarize")
def summarize_video(request: VideoRequest):
    video_id = get_video_id(request.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    transcript = get_transcript(video_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")

    try:
        prompt = f"Summarize this YouTube video:\n\n{transcript}"
        response = model.generate_content(prompt)
        return {"summary": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
