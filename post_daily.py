"""Daily Yoga-pose -> Instagram poster.

Flow:
  1. Pick today's asana from a fixed list (no repeats, correct Hindi names).
  2. Groq (Llama 3) writes: a Hindi one-line benefit, a pose description, Hinglish benefit points.
  3. Pollinations AI draws a minimalist vector illustration; Pillow prints the HINDI text on it.
  4. Image is stored in your public GitHub repo (public URL) -> Instagram Graph API posts it.

SAFE RETRY RULES (so a problem never turns into hammering an API or double-posting):
  * Only temporary problems are retried: network errors, HTTP 408/429/5xx, bad/empty answers.
  * Client errors (401/403/404 ...) stop immediately - e.g. a suspended key is NOT retried.
  * Every step has a hard limit (see MAX_* below) with growing waits (5s, 10s, ...).
  * "Publish to Instagram" is NEVER retried, so one run can never post twice.
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

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
IG_USER_ID = os.environ["IG_USER_ID"]
IG_ACCESS_TOKEN = os.environ["IG_ACCESS_TOKEN"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPOSITORY = os.environ["GITHUB_REPOSITORY"]  # "user/repo" (auto-set in Actions)
GRAPH = "https://graph.instagram.com/v21.0"  # Instagram Login (no Facebook Page needed)

# ---- hard retry limits (total tries, including the first one) ----
MAX_GROQ_TRIES = 3
MAX_IMAGE_TRIES = 3
MAX_GITHUB_TRIES = 3
MAX_IG_CREATE_TRIES = 2
IG_PUBLISH_TRIES = 1  # never retry publishing: avoids double posts
RETRY_STATUS = {408, 429, 500, 502, 503, 504}

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


# ---------------------------------------------------------------- safe retry helpers
class FatalError(Exception):
    """Do not retry - stop the run with a clear message."""


class TransientError(Exception):
    """Temporary problem - may be retried (within the limit)."""


def send(method, url, what, **kw):
    """One HTTP request. Raises TransientError (retryable) or FatalError (stop now)."""
    try:
        r = requests.request(method, url, **kw)
    except (requests.ConnectionError, requests.Timeout) as e:
        raise TransientError(f"{what}: network problem ({e.__class__.__name__})")
    if r.status_code < 400:
        return r
    msg = f"{what}: HTTP {r.status_code}: {r.text[:300]}"
    if r.status_code in RETRY_STATUS:
        raise TransientError(msg)
    raise FatalError(msg)  # 400/401/403/404...: retrying would not help


def retry(fn, what, tries, base_delay=5):
    """Run fn() at most `tries` times. Only TransientError is retried."""
    for i in range(1, tries + 1):
        try:
            return fn()
        except TransientError as e:
            print(f"{what}: try {i}/{tries} failed - {e}")
            if i == tries:
                raise FatalError(f"{what}: giving up after {tries} tries - {e}")
            time.sleep(base_delay * 2 ** (i - 1))  # 5s, 10s, 20s ...


# ---------------------------------------------------------------- Groq text
def get_texts(pose_en, pose_hi):
    """Ask Groq for Hindi one-liner, pose description and Hinglish benefit points."""
    ask = (
        f"Yoga asana: {pose_en} ({pose_hi}).\n"
        "Return JSON with exactly these keys:\n"
        '"benefit_hindi": ONE short sentence in correct, simple, grammatical Hindi written '
        "in Devanagari script only (no English letters, no emoji), max 12 words, saying the "
        "main benefit of this asana. Example style: 'यह आसन पीठ को मज़बूत और मन को शांत करता है।'\n"
        '"pose_description": English, max 35 words, describing the exact body position of this '
        "asana (legs, arms, spine, head, gaze). Describe only the body position - no clothing, "
        "no art style, no background.\n"
        '"benefits_hinglish": a list of 5 short benefit points in Hinglish (Hindi written in '
        "English letters, e.g. 'Pet ki charbi kam karne me madad karta hai'). Each under 70 "
        "characters. Be modest: use 'madad karta hai', never claim to cure any disease.\n"
        '"how_to_hinglish": ONE short line in Hinglish on how to do it safely.'
    )
    url = "https://api.groq.com/openai/v1/chat/completions"

    def attempt():
        r = send(
            "POST",
            url,
            "Groq",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": ask}],
                "temperature": 0.4,
                "response_format": {"type": "json_object"}
            },
            timeout=60,
        )
        try:
            text = r.json()["choices"][0]["message"]["content"]
            d = json.loads(text)
            hi = d["benefit_hindi"].strip()
            pts = [str(p).strip() for p in d["benefits_hinglish"] if str(p).strip()]
            desc = d["pose_description"].strip()
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            raise TransientError(f"Groq: unreadable answer ({e.__class__.__name__})")
        if not DEVANAGARI.search(hi) or LATIN.search(hi) or len(hi) > 110:
            raise TransientError(f"Groq: Hindi line failed checks: {hi!r}")
        if len(pts) < 3 or not desc:
            raise TransientError("Groq: answer incomplete")
        return {
            "benefit_hindi": hi,
            "pose_description": desc,
            "points": pts[:5],
            "how_to": str(d.get("how_to_hinglish", "")).strip(),
        }

    return retry(attempt, "Groq", MAX_GROQ_TRIES)


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
def build_image_prompt(pose_en, description):
    return (
        f"A minimalist, highly aesthetic vector illustration of a person doing the {pose_en} "
        f"yoga pose in modest clothing. {description} "
        "Flat vector style, soft calm pastel colors, clean smooth lines, plain light "
        "background, full body visible and centered in the upper two thirds of the image, "
        "empty space at the bottom. No text, no letters, no watermark, no logo."
    )


def generate_image(prompt):
    """Free image generation via Pollinations AI. Returns a PIL image (RGB, 1080x1350)."""
    full_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{full_prompt}?width={W}&height={H}&nologo=true&model=flux"

    def attempt():
        r = send("GET", url, "Image generation", timeout=180)
        if not r.headers.get("content-type", "").startswith("image"):
            raise TransientError("Image generation: did not return an image")
        try:
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            raise TransientError(f"Image generation: invalid image data - {e}")
        
        if img.size != (W, H):
            img = img.resize((W, H))
        return img

    return retry(attempt, "Image generation", MAX_IMAGE_TRIES, base_delay=10)


def find_hindi_font():
    candidates = (
        glob.glob("/usr/share/fonts/**/NotoSansDevanagari-Bold.ttf", recursive=True)
        + glob.glob("/usr/share/fonts/**/NotoSansDevanagari*Bold*.ttf", recursive=True)
        + glob.glob("/usr/share/fonts/**/FreeSansBold.ttf", recursive=True)
        + glob.glob("/usr/share/fonts/**/Lohit-Devanagari.ttf", recursive=True)
    )
    if not candidates:
        raise FatalError("No Devanagari font found (install fonts-noto-core).")
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
    """Print correct Hindi text on a soft gradient at the bottom of the image."""
    if not features.check("raqm"):
        # Without raqm, Hindi matras render in the wrong order - never post that.
        raise FatalError("Pillow has no raqm support: Hindi text would render incorrectly.")
    font_path = find_hindi_font()
    title_font = ImageFont.truetype(font_path, 110, layout_engine=ImageFont.Layout.RAQM)
    body_font = ImageFont.truetype(font_path, 54, layout_engine=ImageFont.Layout.RAQM)

    # Light illustration background -> dark gradient band keeps the white text readable.
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
    try:  # 404 here just means "file does not exist yet" - that is fine
        existing = requests.get(api, headers=headers, timeout=30)
        if existing.ok:
            payload["sha"] = existing.json()["sha"]
    except requests.RequestException:
        pass
    retry(
        lambda: send("PUT", api, "GitHub upload", headers=headers, json=payload, timeout=60),
        "GitHub upload",
        MAX_GITHUB_TRIES,
    )
    raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/main/{name}"
    for _ in range(10):  # wait (max ~50s) until the URL is actually reachable
        try:
            if requests.head(raw_url, timeout=20).status_code == 200:
                return raw_url
        except requests.RequestException:
            pass
        time.sleep(5)
    raise FatalError("Uploaded image URL is not reachable (is the repo public?)")


def post_to_instagram(image_url, caption):
    r = retry(
        lambda: send(
            "POST",
            f"{GRAPH}/{IG_USER_ID}/media",
            "Instagram create",
            data={"image_url": image_url, "caption": caption, "access_token": IG_ACCESS_TOKEN},
            timeout=60,
        ),
        "Instagram create",
        MAX_IG_CREATE_TRIES,
    )
    container_id = r.json()["id"]

    for _ in range(20):  # wait (max ~100s) for processing
        try:
            s = requests.get(
                f"{GRAPH}/{container_id}",
                params={"fields": "status_code", "access_token": IG_ACCESS_TOKEN},
                timeout=30,
            ).json()
        except (requests.RequestException, ValueError):
            s = {}
        if s.get("status_code") == "FINISHED":
            break
        if s.get("status_code") == "ERROR":
            raise FatalError(f"Instagram processing error: {s}")
        time.sleep(5)

    # Publish exactly ONCE - a retry could create a duplicate post.
    r = retry(
        lambda: send(
            "POST",
            f"{GRAPH}/{IG_USER_ID}/media_publish",
            "Instagram publish",
            data={"creation_id": container_id, "access_token": IG_ACCESS_TOKEN},
            timeout=60,
        ),
        "Instagram publish",
        IG_PUBLISH_TRIES,
    )
    return r.json()["id"]


def main():
    pose_en, pose_hi = todays_pose()
    print("Pose:", pose_en, pose_hi)
    texts = get_texts(pose_en, pose_hi)
    print("Hindi line:", texts["benefit_hindi"])
    prompt = build_image_prompt(pose_en, texts["pose_description"])
    print("Image prompt:", prompt)
    img = add_hindi_text(generate_image(prompt), pose_hi, texts["benefit_hindi"])
    caption = build_caption(pose_en, pose_hi, texts)
    url = upload_to_github(to_jpeg(img))
    print("Image URL:", url)
    post_id = post_to_instagram(url, caption)
    print("Posted! Media ID:", post_id)


if __name__ == "__main__":
    try:
        main()
    except FatalError as e:
        sys.exit(f"STOPPED: {e}")
