"""Daily Yoga-pose -> Instagram poster.

Flow:
  1. Pick today's asana from a fixed list (no repeats, correct Hindi names).
  2. Gemini writes: a Hindi one-line benefit, an image prompt, Hinglish benefit points.
  3. Pollinations draws the pose; Pillow prints the HINDI text on the image
     (AI image models cannot spell Hindi, so we never let them draw the text).
  4. Image is stored in your public GitHub repo (public URL) -> Instagram Graph API posts it.
"""
import base64
import datetime
import glob
import io
import json
import os
import re
import sys
import time
import urllib.parse

import requests
from PIL import Image, ImageDraw, ImageFont, features

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
IG_USER_ID = os.environ["IG_USER_ID"]
IG_ACCESS_TOKEN = os.environ["IG_ACCESS_TOKEN"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPOSITORY = os.environ["GITHUB_REPOSITORY"]  # "user/repo" (auto-set in Actions)
GRAPH = "https://graph.instagram.com/v21.0"  # Instagram Login (no Facebook Page needed)

W, H = 1080, 1350  # Instagram 4:5 portrait

# (English name, Hindi name) - one pose per day, cycles through the list.
POSES = [
    ("Tadasana", "ताड़ासन"),
    ("Vrikshasana", "वृक्षासन"),
    ("Trikonasana", "त्रिकोणासन"),
    ("Bhujangasana", "भुजंगासन"),
    ("Adho Mukha Svanasana", "अधोमुख श्वानासन"),
    ("Balasana", "बालासन"),
    ("Vajrasana", "वज्रासन"),
    ("Padmasana", "पद्मासन"),
    ("Paschimottanasana", "पश्चिमोत्तानासन"),
    ("Dhanurasana", "धनुरासन"),
    ("Setu Bandhasana", "सेतुबंधासन"),
    ("Ardha Matsyendrasana", "अर्धमत्स्येन्द्रासन"),
    ("Virabhadrasana", "वीरभद्रासन"),
    ("Naukasana", "नौकासन"),
    ("Shalabhasana", "शलभासन"),
    ("Matsyasana", "मत्स्यासन"),
    ("Ustrasana", "उष्ट्रासन"),
    ("Gomukhasana", "गोमुखासन"),
    ("Utkatasana", "उत्कटासन"),
    ("Marjaryasana", "मार्जरी आसन"),
    ("Shavasana", "शवासन"),
    ("Sukhasana", "सुखासन"),
    ("Garudasana", "गरुड़ासन"),
    ("Uttanasana", "उत्तानासन"),
    ("Janu Sirsasana", "जानुशीर्षासन"),
    ("Ardha Chandrasana", "अर्धचंद्रासन"),
]

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


def todays_pose():
    idx = int(os.environ.get("POSE_INDEX", datetime.date.today().toordinal())) % len(POSES)
    return POSES[idx]


# ---------------------------------------------------------------- Gemini text
def get_texts(pose_en, pose_hi):
    """Ask Gemini for Hindi one-liner, image prompt and Hinglish benefit points."""
    ask = (
        f"Yoga asana: {pose_en} ({pose_hi}).\n"
        "Return JSON with exactly these keys:\n"
        '"benefit_hindi": ONE short sentence in correct, simple, grammatical Hindi written '
        "in Devanagari script only (no English letters, no emoji), max 12 words, saying the "
        "main benefit of this asana. Example style: 'यह आसन पीठ को मज़बूत और मन को शांत करता है।'\n"
        '"image_prompt": English, max 45 words, describing a calm person doing this exact '
        "asana with correct body position (describe the position of legs, arms, spine and "
        "head), full body visible, clean yoga studio or sunrise outdoors, soft light.\n"
        '"benefits_hinglish": a list of 5 short benefit points in Hinglish (Hindi written in '
        "English letters, e.g. 'Pet ki charbi kam karne me madad karta hai'). Each under 70 "
        "characters. Be modest: use 'madad karta hai', never claim to cure any disease.\n"
        '"how_to_hinglish": ONE short line in Hinglish on how to do it safely.'
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(
                url,
                headers={"x-goog-api-key": GEMINI_API_KEY},
                json={
                    "contents": [{"parts": [{"text": ask}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.4,
                    },
                },
                timeout=60,
            )
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            d = json.loads(text)
            hi = d["benefit_hindi"].strip()
            pts = [str(p).strip() for p in d["benefits_hinglish"] if str(p).strip()]
            if not DEVANAGARI.search(hi) or LATIN.search(hi) or len(hi) > 110:
                raise ValueError(f"Hindi line failed checks: {hi!r}")
            if len(pts) < 3:
                raise ValueError("too few benefit points")
            return {
                "benefit_hindi": hi,
                "image_prompt": d["image_prompt"].strip(),
                "points": pts[:5],
                "how_to": str(d.get("how_to_hinglish", "")).strip(),
            }
        except Exception as e:  # retry on any bad answer
            last_err = e
            print(f"Gemini attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    raise RuntimeError(f"Gemini failed 3 times: {last_err}")


def build_caption(pose_en, pose_hi, t):
    lines = [f"🧘 {pose_hi} ({pose_en})", "", "✨ Fayde (Benefits):"]
    lines += [f"✅ {p}" for p in t["points"]]
    if t["how_to"]:
        lines += ["", f"📝 Kaise kare: {t['how_to']}"]
    lines += [
        "",
        "⚠️ Dhyan rahe: Yoga apne sharir ke hisaab se kare. Koi bimari, chot ya pregnancy ho "
        "to pehle doctor ya yoga trainer se salaah le.",
        "",
        "#yoga #yogaeveryday #yogaindia #asana #yogapractice #healthylifestyle "
        f"#{pose_en.replace(' ', '').lower()}",
        "Made with AI 🤖",
    ]
    return "\n".join(lines)[:2200]


# ---------------------------------------------------------------- image
def generate_image(prompt):
    """Free image generation via Pollinations. Returns a PIL image (RGB)."""
    full = (
        f"{prompt}. Photorealistic yoga photograph, correct human anatomy, one person, "
        "no text, no letters, no watermark, no logo."
    )
    url = (
        "https://image.pollinations.ai/prompt/"
        + urllib.parse.quote(full)
        + f"?width={W}&height={H}&nologo=true"
    )
    for attempt in range(3):
        r = requests.get(url, timeout=180)
        if r.ok and r.headers.get("content-type", "").startswith("image"):
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            if img.size != (W, H):  # make sure it is exactly 4:5
                img = img.resize((W, H))
            return img
        time.sleep(10 * (attempt + 1))
    raise RuntimeError("Image generation failed after 3 attempts")


def find_hindi_font():
    candidates = (
        glob.glob("/usr/share/fonts/**/NotoSansDevanagari-Bold.ttf", recursive=True)
        + glob.glob("/usr/share/fonts/**/NotoSansDevanagari*Bold*.ttf", recursive=True)
        + glob.glob("/usr/share/fonts/**/FreeSansBold.ttf", recursive=True)
        + glob.glob("/usr/share/fonts/**/Lohit-Devanagari.ttf", recursive=True)
    )
    if not candidates:
        raise RuntimeError("No Devanagari font found (install fonts-noto-core).")
    return candidates[0]


def wrap(draw, text, font, max_width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font, language="hi") <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def add_hindi_text(img, pose_hi, benefit_hi):
    """Print correct Hindi text on a dark gradient at the bottom of the image."""
    if not features.check("raqm"):
        # Without raqm, Hindi matras render in the wrong order - never post that.
        raise RuntimeError("Pillow has no raqm support: Hindi text would render incorrectly.")
    font_path = find_hindi_font()
    title_font = ImageFont.truetype(font_path, 110, layout_engine=ImageFont.Layout.RAQM)
    body_font = ImageFont.truetype(font_path, 54, layout_engine=ImageFont.Layout.RAQM)

    # gradient band
    band_h = 520
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for i in range(band_h):
        alpha = int(215 * (i / band_h) ** 1.4)
        od.line([(0, H - band_h + i), (W, H - band_h + i)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    d = ImageDraw.Draw(img)

    body_lines = wrap(d, benefit_hi, body_font, W - 160)
    line_h = 78
    total_h = 140 + 20 + line_h * len(body_lines)
    y = H - 70 - total_h
    d.text((W // 2, y), pose_hi, font=title_font, fill="#FFD369", anchor="ma", language="hi")
    y += 160
    for ln in body_lines:
        d.text((W // 2, y), ln, font=body_font, fill="white", anchor="ma", language="hi")
        y += line_h
    return img.convert("RGB")


def to_jpeg(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


# ---------------------------------------------------------------- GitHub + Instagram
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
    pose_en, pose_hi = todays_pose()
    print("Pose:", pose_en, pose_hi)
    texts = get_texts(pose_en, pose_hi)
    print("Hindi line:", texts["benefit_hindi"])
    img = add_hindi_text(generate_image(texts["image_prompt"]), pose_hi, texts["benefit_hindi"])
    caption = build_caption(pose_en, pose_hi, texts)
    url = upload_to_github(to_jpeg(img))
    print("Image URL:", url)
    post_id = post_to_instagram(url, caption)
    print("Posted! Media ID:", post_id)


if __name__ == "__main__":
    main()
