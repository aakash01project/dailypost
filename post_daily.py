"""Daily AI image -> Instagram poster.

Flow: Gemini writes prompt+caption -> Pollinations makes image -> image is
uploaded to your public GitHub repo (for a public URL) -> Instagram Graph API posts it.
"""
import base64
import datetime
import io
import json
import os
import sys
import time
import urllib.parse

import requests
from PIL import Image

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
IG_USER_ID = os.environ["IG_USER_ID"]
IG_ACCESS_TOKEN = os.environ["IG_ACCESS_TOKEN"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPOSITORY = os.environ["GITHUB_REPOSITORY"]  # "user/repo" (auto-set in Actions)
THEME = os.environ.get("ACCOUNT_THEME", "beautiful fantasy landscapes")
GRAPH = "https://graph.instagram.com/v21.0"  # Instagram Login (no Facebook Page needed)


def get_prompt_and_caption():
    """Ask Gemini for today's image prompt and Instagram caption."""
    today = datetime.date.today().isoformat()
    ask = (
        f"Today is {today}. Account theme: {THEME}. "
        "Create ONE fresh, original idea for an image. Return JSON with keys "
        '"image_prompt" (detailed, visual, under 60 words) and "caption" '
        "(engaging, max 3 sentences, plus 5 relevant hashtags, and end with "
        '"Made with AI").'
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    r = requests.post(
        url,
        headers={"x-goog-api-key": GEMINI_API_KEY},
        json={
            "contents": [{"parts": [{"text": ask}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        },
        timeout=60,
    )
    r.raise_for_status()
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    data = json.loads(text)
    return data["image_prompt"], data["caption"]


def generate_image(prompt):
    """Free image generation via Pollinations. Returns JPEG bytes."""
    url = (
        "https://image.pollinations.ai/prompt/"
        + urllib.parse.quote(prompt)
        + "?width=1080&height=1350&nologo=true"
    )
    for attempt in range(3):
        r = requests.get(url, timeout=180)
        if r.ok and r.headers.get("content-type", "").startswith("image"):
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=92)
            return buf.getvalue()
        time.sleep(10 * (attempt + 1))
    raise RuntimeError("Image generation failed after 3 attempts")


def upload_to_github(jpeg_bytes):
    """Store the image in the repo so Instagram can fetch it from a public URL."""
    name = f"images/{datetime.date.today().isoformat()}.jpg"
    api = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/contents/{name}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    payload = {
        "message": f"Add image {name}",
        "content": base64.b64encode(jpeg_bytes).decode(),
    }
    existing = requests.get(api, headers=headers, timeout=30)
    if existing.ok:
        payload["sha"] = existing.json()["sha"]
    r = requests.put(api, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/main/{name}"
    for _ in range(10):  # wait until the URL is actually reachable
        if requests.head(raw_url, timeout=20).status_code == 200:
            return raw_url
        time.sleep(5)
    raise RuntimeError("Uploaded image URL is not reachable (is the repo public?)")


def post_to_instagram(image_url, caption):
    r = requests.post(
        f"{GRAPH}/{IG_USER_ID}/media",
        data={"image_url": image_url, "caption": caption, "access_token": IG_ACCESS_TOKEN},
        timeout=60,
    )
    if not r.ok:
        sys.exit(f"Container creation failed: {r.text}")
    container_id = r.json()["id"]

    for _ in range(20):  # wait for processing
        s = requests.get(
            f"{GRAPH}/{container_id}",
            params={"fields": "status_code", "access_token": IG_ACCESS_TOKEN},
            timeout=30,
        ).json()
        if s.get("status_code") == "FINISHED":
            break
        if s.get("status_code") == "ERROR":
            sys.exit(f"Instagram processing error: {s}")
        time.sleep(5)

    r = requests.post(
        f"{GRAPH}/{IG_USER_ID}/media_publish",
        data={"creation_id": container_id, "access_token": IG_ACCESS_TOKEN},
        timeout=60,
    )
    if not r.ok:
        sys.exit(f"Publish failed: {r.text}")
    return r.json()["id"]


def main():
    prompt, caption = get_prompt_and_caption()
    print("Prompt:", prompt)
    jpeg = generate_image(prompt)
    url = upload_to_github(jpeg)
    print("Image URL:", url)
    post_id = post_to_instagram(url, caption)
    print("Posted! Media ID:", post_id)


if __name__ == "__main__":
    main()
