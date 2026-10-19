"""Fast version of make_reel_v2.py: same look, much less work for the laptop.

What changed vs v2:
- ffmpeg (fast C code) does the zoom/pan instead of Python resizing every frame.
- Captions/hook are drawn once per change (not once per frame) and laid over the video by ffmpeg.
- Scenes render in parallel, one ffmpeg per scene, then get joined without re-encoding.
- Photos and voices download at the same time; voices are cached, so re-runs skip them.

Usage:  python make_reel_fast.py OUT_DIR            (whole reel)
        python make_reel_fast.py OUT_DIR --scene 0  (one scene only, for quick tests)
"""
import argparse, asyncio, base64, io, itertools, json, os, re, subprocess, time, urllib.parse, urllib.request, wave
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import edge_tts
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1080, 1920, 30
FORMATS = {"9:16": (1080, 1920), "1:1": (1080, 1080), "16:9": (1920, 1080)}

# Free background music: Kevin MacLeod tracks on Wikimedia Commons (CC BY, credit "Kevin MacLeod (incompetech.com)").
MUSIC = {
    "upbeat": ["Hit the Streets v2 (ISRC USUAN1100882).mp3", "Exhilarate (ISRC USUAN1300028).mp3",
               "Back on Track (ISRC USUAN1100426).mp3", "Funin and Sunin (ISRC USUAN1700011).mp3"],
    "calm": ["Fireflies and Stardust (ISRC USUAN1600061).mp3", "Bathed in the Light (ISRC USUAN1100308).mp3",
             "Light Thought var 1 (ISRC USUAN1200006).mp3", "Montauk Point (ISRC USUAN1400013).mp3"],
    "curious": ["Sneaky Adventure (ISRC USUAN1100644).mp3", "Dispersion Relation (ISRC USUAN1100258).mp3",
                "Movement Proposition (ISRC USUAN1100778).mp3", "Sneaky Snitch by Kevin MacLeod.ogg"],
    "dramatic": ["Decisions (ISRC USUAN1100756).mp3", "Heart of Nowhere (ISRC USUAN1400045).mp3",
                 "Long Road Ahead (ISRC USUAN1100588).mp3", "Unlight (ISRC USUAN1100878).mp3"],
    "tech": ["Blipotron (ISRC USUAN1100514).mp3", "BlockMan (ISRC USUAN1100513).mp3", "The Builder (ISRC USUAN1400016).mp3"],
}
MUSIC_VOLUME = 0.12  # quiet enough to keep the voice clear
SS = 2  # zoom source is 2x the frame size, so ffmpeg's whole-pixel panning stays smooth
VOICE = "en-US-AndrewMultilingualNeural"
UA = {"User-Agent": "Mozilla/5.0 (sample-reel demo)"}
WORKERS = max(1, (os.cpu_count() or 2) // 2)  # one scene per physical core
OUT = Path(".")

# Big on-screen text for the first seconds: the thing that stops the scroll.
HOOK_TEXT = "ITS HEART STOPS\nEVERY TIME IT SWIMS"

# (narration, photo search, word the photo's title/tags must contain)
SCENES = [
    ("This animal's heart stops every single time it swims.", "octopus eye close up", "octopus|polpo|pulpo|poulpe|krake"),
    ("Meet the octopus. It doesn't have one heart. It has three.", "octopus underwater", "octopus|polpo|pulpo|poulpe|krake"),
    ("Two of them do just one job: pushing blood through its gills.", "octopus tentacles underwater", "octopus|polpo|pulpo|poulpe|krake"),
    ("The third pumps blood to the rest of the body, and that's the one that shuts off when it swims.",
     "octopus swimming", "octopus|polpo|pulpo|poulpe|krake"),
    ("So the octopus does the smart thing. It crawls instead.", "octopus reef", "octopus|polpo|pulpo|poulpe|krake"),
    ("Oh, and its blood? It's blue. Follow for part two.", "blue octopus", "octopus|polpo|pulpo|poulpe|krake"),
]


def ff(*args):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *map(str, args)], check=True)


# ---------- assets: photos + voice ----------

def get_json(url, tries=3):
    """GET a JSON API, retrying slow or flaky responses (the free photo API sometimes stalls)."""
    for attempt in range(1, tries + 1):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read())
        except Exception as e:
            if attempt == tries:
                raise
            print(f"  search attempt {attempt} failed ({e}), retrying", flush=True)
            time.sleep(3 * attempt)


# titles/tags that usually mean a drawing, diagram or object rather than a real photo
NOT_PHOTOS = ("anatomy", "diagram", "illustration", "drawing", "sketch", "tile", "mosaic", "statue",
              "sculpture", "painting", "engraving", "woodblock", "ukiyo", "cartoon", "logo", "fossil",
              "cthulhu", "pływalnia", "park wodny", "swimming pool", "aquapark", "bowling", "restaurant", "hotel")


BLOCKED_HOSTS = set()  # hosts that answered "429 too many requests" this run


def polite_url(url):
    """Wikimedia asks bots for thumbnails instead of full originals (and rate-limits originals hard)."""
    m = re.match(r"https://upload\.wikimedia\.org/wikipedia/commons/(\w)/(\w\w)/([^/]+)$", url)
    if m:
        return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{m[1]}/{m[2]}/{m[3]}/1280px-{m[3]}"
    return url


TOPIC = ""  # set from the script file; helps the photo judge
VISION_MODEL = os.environ.get("REEL_VISION_MODEL", "")  # e.g. gemma3:27b on Kaggle; empty = no photo check
GOOD, OK_SCORE = 7, 5     # judge scores: stop searching at GOOD, accept the best if it reaches OK_SCORE
PER_SEARCH = 3            # photos downloaded and judged per search
MAX_JUDGED = 12           # per scene, so a hard topic can't take forever


def search_openverse(query):
    q = urllib.parse.urlencode({"q": query, "page_size": 20, "mature": "false",
                                "license_type": "commercial,modification", "size": "large"})
    for r in get_json("https://api.openverse.org/v1/images/?" + q)["results"]:
        tags = " ".join(t["name"] for t in r.get("tags") or [])
        yield {"url": polite_url(r["url"]), "key": r["url"], "title": r.get("title") or "", "text": f"{r.get('title')} {tags}",
               "credit": r.get("attribution") or r.get("title") or r["url"]}


def search_commons(query):
    """Wikimedia Commons directly: every file is freely licensed, and new uploads show up sooner than in Openverse."""
    q = urllib.parse.urlencode({"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
                                "gsrsearch": f"filetype:bitmap {query}", "gsrlimit": 12, "prop": "imageinfo",
                                "iiprop": "url|extmetadata", "iiurlwidth": 1280})
    pages = get_json("https://commons.wikimedia.org/w/api.php?" + q).get("query", {}).get("pages", {})
    for p in sorted(pages.values(), key=lambda p: p.get("index", 99)):
        info = (p.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        title = p["title"].removeprefix("File:").rsplit(".", 1)[0]
        artist = re.sub("<[^>]+>", "", (meta.get("Artist") or {}).get("value", "unknown")).strip()
        lic = (meta.get("LicenseShortName") or {}).get("value", "free license")
        url = info.get("thumburl") or info.get("url")
        if url:
            yield {"url": url, "key": info.get("url", url), "title": title, "text": title,
                   "credit": f'"{title}" by {artist}, {lic}, via Wikimedia Commons'}


def search_nasa(query):
    """NASA's image library: free, mostly public domain. Only worth it for space/science topics, but cheap to ask."""
    q = urllib.parse.urlencode({"q": query, "media_type": "image", "page_size": 10})
    for item in get_json("https://images-api.nasa.gov/search?" + q).get("collection", {}).get("items", [])[:10]:
        data = (item.get("data") or [{}])[0]
        link = next((l["href"] for l in item.get("links", []) if l.get("rel") == "preview"), "")
        if link:  # the preview link is a small thumb; ~medium is a good size for video
            url = link.replace("~thumb.jpg", "~medium.jpg")
            title = data.get("title", "")
            yield {"url": url, "key": url, "title": title, "text": f"{title} {' '.join(data.get('keywords') or [])}",
                   "credit": f'"{title}" by {data.get("center", "NASA")} / NASA (public domain)'}


FLICKR_KEY = os.environ.get("FLICKR_API_KEY", "")
FLICKR_LICENSES = {"4": "CC BY 2.0", "5": "CC BY-SA 2.0", "9": "CC0", "10": "Public Domain"}


def search_flickr(query):
    """Flickr photos under licenses that allow reuse (needs a free API key). People often post event photos same-day."""
    if not FLICKR_KEY:
        return
    q = urllib.parse.urlencode({"method": "flickr.photos.search", "api_key": FLICKR_KEY, "text": query,
                                "license": ",".join(FLICKR_LICENSES), "content_types": 0, "media": "photos",
                                "sort": "relevance", "per_page": 15, "extras": "url_l,url_c,owner_name,license,tags",
                                "format": "json", "nojsoncallback": 1, "safe_search": 1})
    for p in get_json("https://api.flickr.com/services/rest/?" + q).get("photos", {}).get("photo", []):
        url = p.get("url_l") or p.get("url_c")
        if url:
            yield {"url": url, "key": url, "title": p.get("title", ""), "text": f"{p.get('title', '')} {p.get('tags', '')}",
                   "credit": f'"{p.get("title", "")}" by {p.get("ownername", "unknown")}, '
                             f'{FLICKR_LICENSES.get(str(p.get("license")), "free license")}, via Flickr'}


PEXELS_KEY = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_KEY = os.environ.get("PIXABAY_API_KEY", "")
VIDEOS_PER_SEARCH = 2   # video clips are bigger downloads, so fewer per search
VIDEO_BONUS = 0.5       # on equal fit, prefer a moving clip over a still photo
VIDEO_FILES = {}        # url -> downloaded clip


def _orientation():
    return "portrait" if H > W else "landscape" if W > H else "square"


def search_pexels_video(query):
    """Pexels stock video (free key). Pexels gives no titles, so the page slug is used as the description."""
    if not PEXELS_KEY:
        return
    q = urllib.parse.urlencode({"query": query, "per_page": 8, "orientation": _orientation(), "size": "medium"})
    req = urllib.request.Request("https://api.pexels.com/videos/search?" + q, headers={**UA, "Authorization": PEXELS_KEY})
    for v in json.loads(urllib.request.urlopen(req, timeout=30).read()).get("videos", []):
        files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4" and f.get("width")]
        files = [f for f in files if min(f["width"], f["height"]) >= 720] or files
        if not files or v.get("duration", 0) < 3:
            continue
        f = min(files, key=lambda f: abs(min(f["width"], f["height"]) - 1080))  # about 1080p, not 4K
        desc = re.sub(r"^\d+|-\d+$", "", v.get("url", "").rstrip("/").rsplit("/", 1)[-1]).replace("-", " ").strip()
        yield {"url": f["link"], "key": f"pexels:{v['id']}", "title": desc or "pexels video", "text": desc,
               "video": True, "duration": v["duration"], "size": (f["width"], f["height"]),
               "credit": f"Video by {v.get('user', {}).get('name', 'unknown')} on Pexels ({v.get('url', '')})"}


def search_pixabay_video(query):
    """Pixabay stock video (free key)."""
    if not PIXABAY_KEY:
        return
    q = urllib.parse.urlencode({"key": PIXABAY_KEY, "q": query[:100], "per_page": 8, "safesearch": "true"})
    for v in get_json("https://pixabay.com/api/videos/?" + q).get("hits", []):
        f = next((v["videos"][k] for k in ("medium", "large", "small") if v.get("videos", {}).get(k, {}).get("url")), None)
        if not f or v.get("duration", 0) < 3:
            continue
        yield {"url": f["url"], "key": f"pixabay:{v['id']}", "title": v.get("tags", "pixabay video"), "text": v.get("tags", ""),
               "video": True, "duration": v["duration"], "size": (f.get("width", 0), f.get("height", 0)),
               "credit": f"Video by {v.get('user', 'unknown')} on Pixabay ({v.get('pageURL', '')})"}


def video_frame(r):
    """Download a clip once and return a frame from its middle (for the judge)."""
    path = VIDEO_FILES.get(r["url"])
    if not path:
        path = OUT / f"clip_{abs(hash(r['url'])) % 10**8}.mp4"
        path.write_bytes(urllib.request.urlopen(urllib.request.Request(r["url"], headers=UA), timeout=120).read())
        VIDEO_FILES[r["url"]] = path
    frame = path.with_suffix(".jpg")
    if not frame.exists():
        ff("-ss", f"{r.get('duration', 4) / 2:.2f}", "-i", path, "-frames:v", 1, "-q:v", 3, frame)
    return Image.open(frame).convert("RGB")


def judge(img, line):
    """Ask the vision LLM how well a photo fits the narration (0-10). No model -> everything passes."""
    if not VISION_MODEL:
        return 10, ""
    small = img.copy()
    small.thumbnail((768, 768))
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=85)
    prompt = f"""This image may be shown in a short educational video while the narrator says:
"{line}"
Video topic: {TOPIC}
Look carefully and answer three things:
1. "real_photo": true only if it is a real photograph or video frame. false for drawings, paintings,
   illustrations, diagrams, scientific plates, sculptures, vases or other artworks, 3D renders, logos, maps,
   screenshots and text-heavy images.
2. "right_subject": true only if it shows the actual subject of the video. false if it is a different
   kind of thing that only looks or sounds similar (a squid for an octopus, a train car or race car for a
   new sports car, a different species, model or person).
3. "score" 0-10 for how well it fits what the narrator says at that moment
   (10 = exactly what is being talked about, 6 = clearly the right thing, 3 = loosely related, 0 = unrelated).
4. "subject_x" and "subject_y": where the centre of the main subject is (a person's face, the car, the animal),
   in percent from the left edge and from the top edge (50, 50 = the middle). Used to crop around it.
Be strict: a wrong pick looks unprofessional.
Return JSON: {{"real_photo": true/false, "right_subject": true/false, "score": <0-10>, "shows": "<a few words>",
"subject_x": <0-100>, "subject_y": <0-100>}}"""
    body = json.dumps({"model": VISION_MODEL, "prompt": prompt, "images": [base64.b64encode(buf.getvalue()).decode()],
                       "format": "json", "stream": False,
                       "options": {"temperature": 0, "num_ctx": 12288}}).encode()  # same as the writer: no model reload
    try:
        req = urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"})
        data = json.loads(json.loads(urllib.request.urlopen(req, timeout=180).read())["response"])
        score = max(0, min(10, int(float(str(data.get("score", 0)).split("/")[0]))))
        shows = str(data.get("shows", ""))[:60]
        try:  # where the subject is, so a crop keeps it (kept on the image object, saved next to the photo)
            img.info["focus"] = [max(0.0, min(1.0, float(data[k]) / 100)) for k in ("subject_x", "subject_y")]
        except (KeyError, TypeError, ValueError):
            pass
        yes = lambda v: str(v).strip().lower() in ("true", "yes", "1")
        if not yes(data.get("right_subject", True)):
            score, shows = min(score, 2), shows + " [wrong subject]"
        elif not yes(data.get("real_photo", True)):
            score, shows = min(score, 4), shows + " [not a real photo]"
        return score, shows
    except Exception as e:
        print(f"  photo judge failed ({e}); accepting the photo", flush=True)
        return OK_SCORE, ""


def candidates(i, query, musts, used):
    """Downloaded photos for one search (both sources, alternating), passing the cheap title filters."""
    lists = []
    for name, search in (("flickr", search_flickr), ("commons", search_commons), ("openverse", search_openverse),
                         ("nasa", search_nasa)):
        try:
            lists.append(list(search(query)))
        except Exception as e:  # one source failing must not stop the reel
            print(f"photo {i}: {name} search for {query!r} failed ({e})", flush=True)
    clips = []
    for name, search in (("pexels", search_pexels_video), ("pixabay", search_pixabay_video)):
        try:
            clips.append(list(search(query)))
        except Exception as e:
            print(f"photo {i}: {name} video search for {query!r} failed ({e})", flush=True)
    # stock video clips first (if keys are set), then photos
    yield from downloads(i, (r for pair in itertools.zip_longest(*clips) for r in pair if r), used, musts,
                         limit=VIDEOS_PER_SEARCH)
    yield from downloads(i, (r for pair in itertools.zip_longest(*lists) for r in pair if r), used, musts)


PHOTO_POOL = []  # photos found during research (Wikipedia article images, opt-in press photos); tried first
IMG_CACHE = {}   # url -> downloaded image, so research photos aren't fetched again for every scene
REJECTED = set() # judged "not a real photo" or "wrong subject": true for every scene, so never judged again


def downloads(i, results, used, musts=None, limit=PER_SEARCH, min_side=700):
    """Download results that pass the cheap filters (title must mention the subject, unless musts is None)."""
    found = 0
    for r in results:
        text = r.get("text", r.get("title", "")).lower()
        if (r["key"] in used or r["key"] in REJECTED or (musts and not any(m in text for m in musts))
                or any(bad in text for bad in NOT_PHOTOS) or urllib.parse.urlparse(r["url"]).netloc in BLOCKED_HOSTS):
            continue
        try:
            img = IMG_CACHE.get(r["url"])
            if img is None and r.get("video"):
                img = IMG_CACHE[r["url"]] = video_frame(r)
            elif img is None:
                time.sleep(1)  # be polite to image hosts
                data = urllib.request.urlopen(urllib.request.Request(r["url"], headers=UA), timeout=60).read()
                img = IMG_CACHE[r["url"]] = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            if getattr(e, "code", None) == 429:
                BLOCKED_HOSTS.add(urllib.parse.urlparse(r["url"]).netloc)
                print(f"photo {i}: {urllib.parse.urlparse(r['url']).netloc} is rate-limiting us, skipping that host", flush=True)
            continue
        if min(img.size) < (min(min_side, 480) if r.get("video") else min_side):
            continue
        yield r, img
        found += 1
        if found >= limit:
            return


def fetch_photo(i, line, query, must_word, used):
    path = OUT / f"photo{i}.jpg"
    credit_path = OUT / f"photo{i}.txt"
    for done in (path, path.with_suffix(".mp4")):  # already chosen on an earlier attempt
        if done.exists():
            return done, credit_path.read_text(encoding="utf-8")
    musts = must_word.lower().split("|")
    words = query.split()
    searches = [" ".join(words[:n]) for n in range(len(words), 0, -1)]  # "car jumping ramp" -> "car jumping" -> "car"
    searches += [m for m in musts[:1] if m not in searches]
    best, judged = None, 0
    pool = [{"key": p["url"], "text": p.get("title", ""), **p} for p in PHOTO_POOL]
    # research photos first (no title filter: they came from the topic's own pages), then the photo searches
    rounds = [("research photos", lambda: downloads(i, pool, used, None, limit=4, min_side=500))] if pool else []
    rounds += [(q, lambda q=q: candidates(i, q, musts, used)) for q in searches]
    for q, fetch in rounds:
        for r, img in fetch():
            score, shows = judge(img, line)
            judged += 1
            if "[wrong subject]" in shows or "[not a real photo]" in shows:
                REJECTED.add(r["key"])
            kind = "video" if r.get("video") else "photo"
            print(f"photo {i}: {score}/10 {kind} {r['title'][:60]!r}" + (f" (shows: {shows})" if shows else ""), flush=True)
            rank = score + (VIDEO_BONUS if r.get("video") else 0)
            if not best or rank > best[3]:
                best = (score, r, img, rank)
            if score >= GOOD or judged >= MAX_JUDGED:
                break
        if best and (best[0] >= GOOD or judged >= MAX_JUDGED):
            break
    if best and best[0] >= (OK_SCORE if VISION_MODEL else 0):
        score, r, img, _ = best
        if r.get("video"):
            path = path.with_suffix(".mp4")
            path.write_bytes(VIDEO_FILES[r["url"]].read_bytes())
        else:
            img.save(path, quality=95)
        used.add(r["key"])
        credit_path.write_text(r["credit"], encoding="utf-8")
        framing_path(path).write_text(json.dumps({"size": list(img.size), "focus": img.info.get("focus")}))
        print(f"photo {i} ok ({score}/10): {'video ' if r.get('video') else ''}{r['title'][:80]!r} {img.size}", flush=True)
        return path, r["credit"]
    if best:
        print(f"photo {i}: best match only scored {best[0]}/10, not using it", flush=True)
    earlier = sorted(p for p in list(OUT.glob("photo*.jpg")) + list(OUT.glob("photo*.mp4")) if p.stem != path.stem)
    if earlier:  # reuse an earlier photo or clip (it gets a different zoom/crop anyway)
        print(f"photo {i}: no fitting new photo, reusing {earlier[-1].name}", flush=True)
        path = path.with_suffix(earlier[-1].suffix)
        path.write_bytes(earlier[-1].read_bytes())
        credit = earlier[-1].with_suffix(".txt").read_text(encoding="utf-8")
        credit_path.write_text(credit, encoding="utf-8")
        if framing_path(earlier[-1]).exists():
            framing_path(path).write_text(framing_path(earlier[-1]).read_text())
        return path, credit
    # very last resort: a plain dark background, so the reel still gets made
    print(f"photo {i}: no photo at all, using a plain background", flush=True)
    bg = Image.new("RGB", (W, H), (18, 22, 34))
    ImageDraw.Draw(bg).ellipse([W * 0.13, H / 2 - W * 0.37, W * 0.87, H / 2 + W * 0.37], fill=(40, 52, 84))
    bg.filter(ImageFilter.GaussianBlur(160)).save(path, quality=95)
    credit_path.write_text("plain background", encoding="utf-8")
    return path, "plain background"


def fetch_photos():
    used = set()
    return [fetch_photo(i, line, q, m, used) for i, (line, q, m) in enumerate(SCENES)]


async def speak(i, text):
    """Returns (wav path, [(start_s, end_s, word), ...]); reuses the cached voice if the text is unchanged."""
    mp3, wav, meta = OUT / f"voice{i}.mp3", OUT / f"voice{i}.wav", OUT / f"voice{i}.json"
    if wav.exists() and meta.exists():
        cached = json.loads(meta.read_text(encoding="utf-8"))
        if cached["text"] == text and cached["voice"] == VOICE:
            return wav, [tuple(w) for w in cached["words"]]
    words = []
    comm = edge_tts.Communicate(text, VOICE, rate="+10%", boundary="WordBoundary")
    with open(mp3, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                s = chunk["offset"] / 1e7
                words.append((s, s + chunk["duration"] / 1e7, chunk["text"]))
    ff("-i", mp3, "-ar", 24000, "-ac", 1, wav)
    meta.write_text(json.dumps({"text": text, "voice": VOICE, "words": words}), encoding="utf-8")
    return wav, words


def fetch_music(mood, seed=""):
    """Download one track for the mood from Wikimedia Commons. Returns (path, credit) or (None, '')."""
    tracks = MUSIC.get(mood)
    if not tracks:
        return None, ""
    order = sorted(tracks, key=lambda t: hash((seed, t)) % 1000)  # varies by topic, stable within a reel
    for title in order:
        try:
            q = urllib.parse.urlencode({"action": "query", "format": "json", "titles": "File:" + title,
                                        "prop": "imageinfo", "iiprop": "url"})
            pages = get_json("https://commons.wikimedia.org/w/api.php?" + q)["query"]["pages"]
            url = next(iter(pages.values()))["imageinfo"][0]["url"]
            path = OUT / ("music" + Path(url).suffix)
            path.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())
            name = re.sub(r"\s*\(ISRC.*|\s*by Kevin MacLeod|\.\w+$", "", title)
            print(f"music: {name} ({mood})", flush=True)
            return path, f'Music: "{name}" by Kevin MacLeod (incompetech.com), CC BY, via Wikimedia Commons'
        except Exception as e:
            print(f"music: could not get {title!r} ({e})", flush=True)
    return None, ""


async def gather_assets():
    photos_task = asyncio.to_thread(fetch_photos)
    voices_task = asyncio.gather(*(speak(i, text) for i, (text, _, _) in enumerate(SCENES)))
    return await asyncio.gather(photos_task, voices_task)


# ---------- text overlays (drawn once per change, not per frame) ----------

@lru_cache(maxsize=None)
def load_font(size):
    for name in (os.environ.get("REEL_FONT", ""), "C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialbd.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):  # last one: Linux, e.g. Kaggle
        if not name:
            continue
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def caption_index(words, t):
    idx = next((k for k, (s, e, _) in enumerate(words) if s <= t < e + 0.15), None)
    if idx is None:
        idx = max((k for k, (s, _, _) in enumerate(words) if s <= t), default=None)
    return idx


def hook_font_size(t):
    """Hook headline starts big and snaps into place over 0.18 s; gone after 2.6 s."""
    if t > 2.6:
        return None
    return int(118 * (1.25 - 0.25 * min(1.0, t / 0.18)))


def draw_caption(img, words, idx):
    """Current 3-word group; the spoken word is yellow and slightly bigger."""
    g0 = idx - idx % 3
    texts = [w[2].upper() for w in words[g0:g0 + 3]]
    d = ImageDraw.Draw(img)
    size = 105
    while True:
        fonts = [load_font(int(size * (1.15 if g0 + k == idx else 1))) for k in range(len(texts))]
        widths = [d.textlength(x, font=f) for x, f in zip(texts, fonts)]
        total = sum(widths) + 28 * (len(texts) - 1)
        if total <= W - 110 or size < 50:
            break
        size -= 6
    x, base = (W - total) / 2, H * 0.70
    for k, (txt, f, w) in enumerate(zip(texts, fonts, widths)):
        hot = g0 + k == idx
        d.text((x, base), txt, font=f, anchor="ls", fill=(255, 222, 0) if hot else (255, 255, 255),
               stroke_width=9, stroke_fill=(0, 0, 0))
        x += w + 28


def draw_hook(img, font_size):
    d = ImageDraw.Draw(img)
    widest = max(HOOK_TEXT.split("\n"), key=lambda line: d.textlength(line, font=load_font(font_size)))
    while font_size > 40 and d.textlength(widest, font=load_font(font_size)) > W - 90:
        font_size -= 4  # keep the pop-in inside the frame
    font = load_font(font_size)
    y = H * 0.16
    for k, line in enumerate(HOOK_TEXT.split("\n")):
        w = d.textlength(line, font=font)
        pad = 22
        d.rectangle([(W - w) / 2 - pad, y - 8, (W + w) / 2 + pad, y + font.size + 14],
                    fill=(220, 20, 40) if k == 0 else (255, 255, 255))
        d.text(((W - w) / 2, y), line, font=font, fill=(255, 255, 255) if k == 0 else (0, 0, 0))
        y += font.size + 34


def build_overlay_track(si, words, dur, work):
    """Writes one transparent PNG per distinct text state plus an ffconcat timeline for ffmpeg."""
    cuts = {0.0, dur}
    for s, e, _ in words:
        cuts |= {s, e + 0.15}
    if si == 0:
        cuts |= {k / FPS for k in range(7)} | {2.6}
    cuts = sorted(c for c in cuts if 0 <= c <= dur)

    timeline = []  # [state, seconds]
    for a, b in zip(cuts, cuts[1:]):
        if b - a < 1e-4:
            continue
        t = a + 1e-4
        state = (caption_index(words, t), hook_font_size(t) if si == 0 else None)
        if timeline and timeline[-1][0] == state:
            timeline[-1][1] += b - a
        else:
            timeline.append([state, b - a])

    files = {}
    for state, _ in timeline:
        if state in files:
            continue
        idx, hook_size = state
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if hook_size:
            draw_hook(img, hook_size)
        if idx is not None:
            draw_caption(img, words, idx)
        path = work / f"s{si}_ov{len(files)}.png"
        img.save(path, compress_level=1)
        files[state] = path

    lines = ["ffconcat version 1.0"]
    for state, secs in timeline:
        lines += [f"file '{files[state].as_posix()}'", f"duration {secs:.4f}"]
    lines.append(f"file '{files[timeline[-1][0]].as_posix()}'")  # concat demuxer needs the last file twice
    track = work / f"s{si}_overlay.ffconcat"
    track.write_text("\n".join(lines), encoding="utf-8")
    return track


# ---------- background photo + zoom ----------

FIT_BELOW = 0.6   # if filling the frame would keep less than this share of the photo, show it whole instead
FIT_SIZE = 0.88   # a whole photo fills this share of the canvas, so the zoom barely touches its edges
FIT_ZOOM = 0.4    # zoom amount for whole photos, compared with full-frame photos
FIT_SHAPE = 4 / 3 # a very wide photo is trimmed to this shape (around its subject) so it shows bigger; 16:9 keeps 75%


def framing_path(photo):
    return Path(photo).with_name(Path(photo).stem + "_framing.json")


def framing(photo):
    """(width, height, focus) saved when the photo was chosen; focus is the subject's centre as 0-1 fractions."""
    try:
        f = json.loads(framing_path(photo).read_text())
        return f["size"][0], f["size"][1], f.get("focus")
    except (OSError, ValueError, KeyError, TypeError):
        return None, None, None


def kept_share(width, height):
    """How much of a width x height photo stays visible if it is cropped to fill the frame."""
    a, fa = width / height, W / H
    return min(a / fa, fa / a)


def cover_crop(img, tw, th, focus=None):
    """Scale to fill tw x th, then crop around the subject (focus) instead of always the middle."""
    s = max(tw / img.width, th / img.height)
    img = img.resize((max(tw, round(img.width * s)), max(th, round(img.height * s))), Image.LANCZOS)
    fx, fy = focus or (0.5, 0.45)
    left = min(max(0, round(fx * img.width - tw / 2)), img.width - tw)
    top = min(max(0, round(fy * img.height - th / 2)), img.height - th)
    return img.crop((left, top, left + tw, top + th))


def trim_bars(img, focus=None):
    """Cut off black bars baked into a photo (video screenshots often have them). Returns (image, focus)."""
    mask = img.convert("L").point(lambda v: 255 if v > 24 else 0)
    box = mask.getbbox()
    if not box or (box[2] - box[0]) * (box[3] - box[1]) < 0.5 * img.width * img.height or box == (0, 0, *img.size):
        return img, focus
    if focus:
        focus = [min(1, max(0, (focus[0] * img.width - box[0]) / (box[2] - box[0]))),
                 min(1, max(0, (focus[1] * img.height - box[1]) / (box[3] - box[1])))]
    return img.crop(box), focus


def prep_image(path, out_path):
    """Make the scene's background at SS x 1.15 the frame (room to zoom and pan), with the dark caption vignette.
    Photos close to the frame's shape fill it, cropped around the subject. Much wider (or taller) photos are shown
    nearly whole, on a blurred and darkened copy of themselves, so a wide news photo isn't cut down to a sliver.
    Returns "fit" or "fill"."""
    img = Image.open(path).convert("RGB")
    tw, th = int(W * 1.15 * SS), int(H * 1.15 * SS)
    _, _, focus = framing(path)
    img, focus = trim_bars(img, focus)
    if kept_share(*img.size) >= FIT_BELOW:
        mode, img = "fill", cover_crop(img, tw, th, focus)
    else:
        mode = "fit"
        back = cover_crop(img, tw // 8, th // 8).filter(ImageFilter.GaussianBlur(6))
        a = img.width / img.height
        if W < H and a > FIT_SHAPE:  # wide photo in a tall reel: trim the sides around the subject
            cw = round(img.height * FIT_SHAPE)
            img = cover_crop(img, cw, img.height, focus)
        elif W > H and a < 1 / FIT_SHAPE:  # tall photo in a wide video: trim top and bottom
            ch = round(img.width * FIT_SHAPE)
            img = cover_crop(img, img.width, ch, focus)
        back = Image.eval(back, lambda v: int(v * 0.55)).resize((tw, th), Image.BILINEAR)
        s = min(tw * FIT_SIZE / img.width, th * FIT_SIZE / img.height)
        front = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
        x, y = (tw - front.width) // 2, round((th - front.height) * 0.45)  # a little above the middle: captions sit low
        shadow = Image.new("L", (tw // 4, th // 4), 0)
        ImageDraw.Draw(shadow).rectangle([x // 4, (y + 8 * SS) // 4, (x + front.width) // 4, (y + front.height + 8 * SS) // 4],
                                         fill=150)
        shadow = shadow.filter(ImageFilter.GaussianBlur(10)).resize((tw, th), Image.BILINEAR)
        back = Image.composite(Image.new("RGB", (tw, th), (0, 0, 0)), back, shadow)
        corners = Image.new("L", front.size, 0)
        ImageDraw.Draw(corners).rounded_rectangle([0, 0, front.width - 1, front.height - 1], radius=18 * SS, fill=255)
        back.paste(front, (x, y), corners)
        img = back
    # blur the vignette mask at quarter size (much faster), then scale it up
    small = Image.new("L", (tw // 4, th // 4), 0)
    d = ImageDraw.Draw(small)
    d.rectangle([0, int(small.height * 0.58), small.width, small.height], fill=140)
    d.rectangle([0, 0, small.width, int(small.height * 0.12)], fill=70)
    mask = small.filter(ImageFilter.GaussianBlur(110 * SS / 4)).resize(img.size, Image.BILINEAR)
    Image.composite(Image.new("RGB", img.size, (0, 0, 0)), img, mask).save(out_path, quality=95)
    return mode


def zoom_filter(si, n, gentle=False):
    """Hook scene punches in fast, others alternate slow zoom in/out with a drift.
    gentle (a whole photo on a blurred backdrop): smaller zoom, so the photo's edges stay in view."""
    p = f"(on/{max(1, n - 1)})"
    k = FIT_ZOOM if gentle else 1
    if si == 0:
        z = f"(1+{0.22 * k:.3f}*(1-(1-{p})*(1-{p})))"
    elif si % 2 == 0:
        z = f"(1+{0.12 * k:.3f}*{p})"
    else:
        z = f"(1+{0.12 * k:.3f}*(1-{p}))"
    drift = 1 if si % 2 else -1
    return (f"zoompan=z='{1.15 / 1.1:.6f}*{z}'"
            f":x='(iw-iw/zoom)*(0.5+{0.25 * drift}*({p}-0.5))'"
            f":y='(ih-ih/zoom)*0.45'"
            f":d={n}:s={W}x{H}:fps={FPS}")


def vignette_png(work):
    """Transparent overlay that darkens the caption area (video clips don't get the photo vignette baked in)."""
    path = work / f"vignette_{W}x{H}.png"
    if not path.exists():
        small = Image.new("L", (W // 4, H // 4), 0)
        d = ImageDraw.Draw(small)
        d.rectangle([0, int(small.height * 0.58), small.width, small.height], fill=140)
        d.rectangle([0, 0, small.width, int(small.height * 0.12)], fill=70)
        alpha = small.filter(ImageFilter.GaussianBlur(110 / 4)).resize((W, H), Image.BILINEAR)
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        img.putalpha(alpha)
        img.save(path)
    return path


def render_scene(si, photo, words, dur, work):
    t0 = time.perf_counter()
    n = int(round(dur * FPS))
    track = build_overlay_track(si, words, dur, work)
    flash = ",fade=t=in:s=0:n=5:color=white" if si > 0 else ""  # white flash on each cut
    seg = work / f"seg{si}.mp4"
    if Path(photo).suffix == ".mp4":  # stock video clip: loop if too short, darken the caption area
        cw, ch, _ = framing(photo)
        if cw and kept_share(cw, ch) < FIT_BELOW:  # e.g. a 16:9 clip in a 9:16 reel: show it whole on a blurred copy
            fw, fh = int(W * FIT_SIZE) // 2 * 2, int(H * FIT_SIZE) // 2 * 2
            base = (f"[0:v]fps={FPS},split[a][b];"
                    f"[a]scale={W // 8}:{H // 8}:force_original_aspect_ratio=increase,crop={W // 8}:{H // 8},"
                    f"boxblur=6,eq=brightness=-0.3,scale={W}:{H},setsar=1[back];"
                    f"[b]scale={fw}:{fh}:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1[front];"
                    f"[back][front]overlay=(W-w)/2:(H-h)*0.45{flash}[bg];")
        else:  # close to the frame's shape: fill it
            base = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},setsar=1{flash}[bg];"
        graph = (base + f"[1:v]format=rgba[vg];[bg][vg]overlay=0:0[bgv];"
                 f"[2:v]format=rgba[ov];"
                 f"[bgv][ov]overlay=0:0:format=auto,scale=out_range=tv,format=yuv420p[v]")
        ff("-stream_loop", -1, "-i", photo, "-loop", 1, "-i", vignette_png(work), "-f", "concat", "-safe", 0, "-i", track,
           "-filter_complex", graph, "-map", "[v]", "-frames:v", n, "-r", FPS,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", 20, "-threads", 2, "-an",
           "-color_range", "tv", "-colorspace", "bt470bg", seg)  # same colour tags as photo scenes, so joins are clean
    else:
        bg = work / f"s{si}_bg.jpg"
        mode = prep_image(photo, bg)
        graph = (f"[0:v]{zoom_filter(si, n, gentle=mode == 'fit')}{flash}[bg];"
                 f"[1:v]format=rgba[ov];"
                 f"[bg][ov]overlay=0:0:format=auto,scale=out_range=tv,format=yuv420p[v]")  # standard range for phones
        ff("-i", bg, "-f", "concat", "-safe", 0, "-i", track,
           "-filter_complex", graph, "-map", "[v]", "-frames:v", n, "-r", FPS,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", 20, "-threads", 2, "-an", seg)
    print(f"scene {si} rendered ({dur:.1f}s of video in {time.perf_counter() - t0:.1f}s)", flush=True)
    return seg


# ---------- main ----------

def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--scene", type=int, help="render only this scene (quick test)")
    ap.add_argument("--script", help='JSON file: {"hook": "...", "scenes": [[narration, photo search, must words], ...]}')
    ap.add_argument("--name", default="sample_reel_fast", help="output video name, without .mp4")
    ap.add_argument("--format", choices=list(FORMATS), help="9:16 (default), 1:1 or 16:9")
    ap.add_argument("--voice", help="an edge-tts voice, e.g. en-IN-PrabhatNeural")
    ap.add_argument("--music", help="off, upbeat, calm, curious, dramatic or tech")
    args = ap.parse_args()
    global W, H, VOICE
    data = {}
    if args.script:
        global HOOK_TEXT, SCENES, TOPIC, PHOTO_POOL
        data = json.loads(Path(args.script).read_text(encoding="utf-8-sig"))  # tolerate Notepad's BOM
        HOOK_TEXT, SCENES = data["hook"], [tuple(s[:3]) for s in data["scenes"]]
        TOPIC = data.get("topic", "")
        PHOTO_POOL = [p for p in data.get("photo_pool", []) if p.get("url")]
    W, H = FORMATS[args.format or data.get("format") or "9:16"]
    VOICE = args.voice or data.get("voice") or VOICE
    music_mood = (args.music or data.get("music") or "off").lower()
    OUT = Path(args.out_dir).resolve()  # absolute: ffconcat resolves relative paths from its own folder
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "work"
    work.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    print(f"format {W}x{H}, voice {VOICE}, music {music_mood}", flush=True)
    photos, voices = asyncio.run(gather_assets())
    music, music_credit = fetch_music(music_mood, TOPIC) if music_mood != "off" else (None, "")
    print(f"assets ready in {time.perf_counter() - t0:.1f}s", flush=True)
    (OUT / "credits.txt").write_text("\n".join([c for _, c in photos] + ([music_credit] if music else [])),
                                     encoding="utf-8")

    pause = 0.15
    durations = []
    for wav, _ in voices:
        with wave.open(str(wav)) as w:
            durations.append(w.getnframes() / w.getframerate() + pause)

    jobs = [(si, photos[si][0], voices[si][1], durations[si], work) for si in range(len(SCENES))]
    if args.scene is not None:
        jobs = [jobs[args.scene]]

    t1 = time.perf_counter()
    with ThreadPoolExecutor(WORKERS) as pool:
        segs = list(pool.map(lambda j: render_scene(*j), jobs))
    print(f"scenes rendered in {time.perf_counter() - t1:.1f}s using {WORKERS} at a time", flush=True)
    if args.scene is not None:
        print("DONE (test scene only)", segs[0])
        return

    # voice track: each scene's voice followed by a short pause
    full_wav = work / "voice_full.wav"
    with wave.open(str(full_wav), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(24000)
        for wav, _ in voices:
            with wave.open(str(wav)) as w:
                out.writeframes(w.readframes(w.getnframes()))
                out.writeframes(b"\x00\x00" * int(24000 * pause))

    # join scene videos without re-encoding, add the voice
    seg_list = work / "segments.txt"
    seg_list.write_text("\n".join(f"file '{s.as_posix()}'" for s in segs), encoding="utf-8")
    video = OUT / f"{args.name}.mp4"
    if music:  # music loops under the voice, quietly, and fades out at the end
        total = sum(durations)
        mix = (f"[2:a]volume={MUSIC_VOLUME},afade=t=in:d=0.5,afade=t=out:st={max(0, total - 1.5):.2f}:d=1.5[m];"
               f"[1:a][m]amix=inputs=2:duration=first:normalize=0[a]")
        ff("-f", "concat", "-safe", 0, "-i", seg_list, "-i", full_wav, "-stream_loop", -1, "-i", music,
           "-filter_complex", mix, "-map", "0:v", "-map", "[a]",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", video)
    else:
        ff("-f", "concat", "-safe", 0, "-i", seg_list, "-i", full_wav,
           "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", video)
    print(f"DONE {video} ({sum(durations):.1f}s reel) in {time.perf_counter() - t0:.1f}s total")


if __name__ == "__main__":
    main()
