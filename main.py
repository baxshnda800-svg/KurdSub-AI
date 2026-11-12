import os
import json
import re
import mimetypes
import base64
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

GEMINI_API_KEY = os.getenv("AQ.Ab8RN6KnHkoE_kLAToFQXRUardSZdG_RYRsyvMfWPAPMJmKKrw")
# Google recommends Gemini 3.8 Flash for new projects; access to some older 2.5 models is restricted for new users.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_BYTES = 18 * 1024 * 1024
GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "source_language": {"type": "STRING"},
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start": {"type": "NUMBER"},
                    "end": {"type": "NUMBER"},
                    "original": {"type": "STRING"},
                    "translated": {"type": "STRING"},
                },
                "required": ["start", "end", "original", "translated"],
            },
        },
    },
    "required": ["source_language", "segments"],
}

app = FastAPI(title="KurdSub AI")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def home():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {"ok": True, "configured": bool(os.getenv("GEMINI_API_KEY"))}


def clean_json(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def extract_interaction_text(payload: dict) -> str:
    """Extract text from the REST Interactions API response."""
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks = []
    for step in payload.get("steps", []):
        if step.get("type") != "model_output":
            continue
        for part in step.get("content", []):
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "\n".join(chunks)


def validate_youtube_url(value: str) -> str:
    value = (value or "").strip()
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower()
    except ValueError:
        host = ""
        parsed = None
    valid_host = host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")
    if not parsed or parsed.scheme not in ("https", "http") or not valid_host:
        raise HTTPException(status_code=400, detail="تەنیا لینکی ڤیدیۆی گشتیی YouTube دابنێ.")
    path = parsed.path or ""
    has_video_id = bool(parsed.query and re.search(r"(?:^|&)v=[\w-]{6,}(?:&|$)", parsed.query))
    has_video_id = has_video_id or (host == "youtu.be" and len(path.strip("/").split("/")) == 1 and len(path.strip("/")) >= 6)
    has_video_id = has_video_id or bool(re.match(r"^/(?:shorts|embed|live)/[\w-]{6,}", path))
    if not has_video_id:
        raise HTTPException(status_code=400, detail="لینکەکە وەک لینکی ڤیدیۆی YouTube دیار نییە.")
    return value


def normalize_segments(parsed: dict, target_language: str):
    raw_segments = parsed.get("segments", [])
    result = []
    for segment in raw_segments:
        try:
            start = max(0, int(float(segment.get("start", 0)) * 1000))
            end = max(start + 500, int(float(segment.get("end", 0)) * 1000))
        except (TypeError, ValueError):
            continue
        original = str(segment.get("original", "")).strip()
        translated = str(segment.get("translated", "")).strip()
        if original or translated:
            result.append({
                "start_ms": start,
                "end_ms": end,
                "original": original,
                "text": translated,
            })
    if not result:
        raise HTTPException(status_code=502, detail="Gemini ژێرنووسێکی دروستی نەگەڕاندەوە. تکایە جارێکی تر هەوڵ بدەرەوە.")
    return {
        "language": parsed.get("source_language", ""),
        "target_language": target_language,
        "segments": result,
    }


def build_prompt(source_language: str, target_language: str) -> str:
    source_hint = "خۆکارانە زمانی قسەکردن بناسەوە"
    if source_language and source_language != "auto":
        source_hint = f"زمانی سەرچاوە {source_language} ـە؛ بەڵام ئەگەر جیاواز بوو، زمانی ڕاستەقینە بناسەوە."
    return f"""You are a professional multilingual speech transcriber and subtitle translator.
Analyze the attached media. {source_hint}
Translate every spoken dialogue segment into this target language: {target_language}.
Return ONLY valid JSON matching this shape:
{{"source_language":"detected language name","segments":[{{"start":0.0,"end":2.4,"original":"spoken words in the original language","translated":"faithful translation in {target_language}"}}]}}
Rules:
- Include all audible spoken dialogue in chronological order; do not summarize or omit lines.
- Keep subtitle segments concise and natural, normally one sentence or short phrase per segment.
- Timestamps are seconds from the beginning of the media. Use the media timeline and be as accurate as possible; never invent precise timing when uncertain.
- Do not invent speech. If a portion is unintelligible, mark only that portion as [unintelligible] in original and translate it appropriately.
- Preserve names, meaning, emotion, and conversational tone.
- The original field must remain in the detected source language. The translated field must be entirely in {target_language}.
- Do not add markdown fences or commentary.
"""


@app.post("/api/process")
async def process(
    file: UploadFile | None = File(None),
    youtube_url: str = Form(""),
    source_language: str = Form("auto"),
    target_language: str = Form("Kurdish Sorani"),
):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="کلیلی Gemini API دانەنراوە. تکایە GEMINI_API_KEY لە Environment ـی سێرڤەر دابنێ.")

    target_language = (target_language or "Kurdish Sorani").strip()[:80]
    prompt = build_prompt(source_language, target_language)
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            if youtube_url.strip():
                # Gemini's public-YouTube URL input is a preview feature.
                safe_url = validate_youtube_url(youtube_url)
                request_body = {
                    "model": os.getenv("GEMINI_MODEL", GEMINI_MODEL),
                    "input": [
                        {"type": "text", "text": prompt},
                        {"type": "video", "uri": safe_url},
                    ],
                    "response_format": [{
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": GEMINI_SCHEMA,
                    }],
                }
                response = await client.post(
                    "https://generativelanguage.googleapis.com/v1beta/interactions",
                    headers=headers,
                    json=request_body,
                )
                if response.status_code >= 400:
                    try:
                        err = response.json().get("error", {}).get("message", response.text)
                    except Exception:
                        err = response.text
                    raise HTTPException(status_code=502, detail=f"هەڵەی Gemini لە لینکی YouTube: {str(err)[:450]}")
                text = extract_interaction_text(response.json())
            else:
                if file is None:
                    raise HTTPException(status_code=400, detail="تکایە فایلێک باربکە یان لینکی YouTube دابنێ.")
                data = await file.read()
                if not data:
                    raise HTTPException(status_code=400, detail="فایلەکە بەتاڵە.")
                if len(data) > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="قەبارەی فایل زیاترە لە 18MB. تکایە فایلەکە بچووک بکەرەوە یان بە پارچەی بچووک دایبەزێنە.")

                filename = file.filename or "media"
                mime_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
                if mime_type == "video/quicktime":
                    mime_type = "video/mov"
                if not mime_type.startswith(("audio/", "video/")):
                    raise HTTPException(status_code=415, detail="تەنیا فایلی دەنگ یان ڤیدیۆ پشتگیری دەکرێت.")

                request_body = {
                    "contents": [{"parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(data).decode("ascii")}},
                    ]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "responseMimeType": "application/json",
                        "responseSchema": GEMINI_SCHEMA,
                    },
                }
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{os.getenv('GEMINI_MODEL', GEMINI_MODEL)}:generateContent"
                response = await client.post(endpoint, headers=headers, json=request_body)
                if response.status_code >= 400:
                    try:
                        err = response.json().get("error", {}).get("message", response.text)
                    except Exception:
                        err = response.text
                    raise HTTPException(status_code=502, detail=f"هەڵەی Gemini API: {str(err)[:450]}")
                payload = response.json()
                text = "".join(
                    part.get("text", "")
                    for part in payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    if "text" in part
                )

        parsed = clean_json(text)
        return normalize_segments(parsed, target_language)

    except HTTPException:
        raise
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="پرۆسەکردنی میدیاکە زۆر درێژ بوو. تکایە کلیپێکی کورتتر بەکاربهێنە.") from exc
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail="Gemini وەڵامێکی دروستی ژێرنووسی نەگەڕاندەوە. تکایە جارێکی تر هەوڵ بدەرەوە.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="پەیوەندی بە Gemini API ـەوە سەرکەوتوو نەبوو. تکایە دواتر هەوڵ بدەرەوە.") from exc
