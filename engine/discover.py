"""Discover (runs on your laptop, no GPU): read the latest headlines from news feeds, group the same story told by
many sites into one story, and score each story out of 100 for how good a reel it would make.

  Popular now  (40)  how many news sites cover it (25) + whether people are searching for it on Google Trends (15)
  Growing fast (20)  how many sites wrote about it in the last 3 hours / since earlier scans
  Reel potential (30)  a small free AI on this laptop (Ollama, llama3.2) judges how watchable and shareable it is
  Fresh (10)  newer stories score higher

Reddit is not used: it blocks programs (403 / 429).
"""
import email.utils, html, json, math, os, re, shutil, subprocess, time, urllib.parse, urllib.request, uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

from research import _get

GN = "https://news.google.com/rss/headlines/section/topic/{}?hl=en-IN&gl=IN&ceid=IN:en"
FEEDS = [  # (name, url). Google News pages already group each story with the other sites covering it
    ("Google News: top", "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"),
    ("Google News: world", GN.format("WORLD")),
    ("Google News: India", GN.format("NATION")),
    ("Google News: business", GN.format("BUSINESS")),
    ("Google News: technology", GN.format("TECHNOLOGY")),
    ("Google News: entertainment", GN.format("ENTERTAINMENT")),
    ("Google News: sports", GN.format("SPORTS")),
    ("Google News: science", GN.format("SCIENCE")),
    ("NDTV", "https://feeds.feedburner.com/ndtvnews-top-stories"),
    ("The Hindu", "https://www.thehindu.com/news/national/feeder/default.rss"),
    ("Indian Express", "https://indianexpress.com/feed/"),
    ("Times of India", "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
    ("BBC", "https://feeds.bbci.co.uk/news/world/rss.xml"),
]
TRENDS = [("India", "https://trends.google.com/trending/rss?geo=IN"), ("US", "https://trends.google.com/trending/rss?geo=US")]
HT = "{https://trends.google.com/trending/rss}"

MAX_AGE_H = 24        # older stories drop off the list
RECENT_H = 3          # "growing fast" looks at this many hours
OLLAMA = "http://127.0.0.1:11434"
AI_MODEL = "llama3.2"
AI_BATCH = 10         # stories per AI call
AI_MAX_NEW = 40       # at most this many new stories are rated per scan (the most popular first)

STOP = set("""a an the and or but if of to in on at by for with from into onto over under about after before amid among
as is are was were be been being has have had do does did will would can could may might must shall should not no
this that these those it its it's he she they them his her their our we you your i me my who whom whose which what
when where why how than then there here also just more most less least very much many some any all each every both
says said say tells told report reports reported according watch video live updates update latest news today now
amid against new first last next year years day days week weeks month months time times big top full know
india indian india's us u.s uk world global govt government official officials people man woman men women
check details explained explainer here's what's why how photos pics breaking exclusive opinion analysis""".split())


# ---------- reading feeds ----------

def _text(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def _when(s):
    try:
        return email.utils.parsedate_to_datetime(s).timestamp()
    except Exception:
        return None


def site_of(url):
    host = urllib.parse.urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def read_feed(name, url):
    root = ET.fromstring(_get(url, timeout=20, tries=2))
    out = []
    for it in root.iter("item"):
        title, link = _text(it.findtext("title")), (it.findtext("link") or "").strip()
        if not title or not link:
            continue
        google = "news.google.com" in link
        source = _text(it.findtext("source")) if google else name
        if google and source and title.endswith(" - " + source):
            title = title[: -len(source) - 3].strip()
        title = re.sub(r"\s+\|\s+[^|]{2,40}$", "", title) if source and title.endswith(source) else title
        desc = it.findtext("description") or ""
        # Google News lists the other sites covering the same story inside the description
        cluster = [_text(s) for s in re.findall(r'<font color="#6f6f6f">([^<]+)</font>', desc)] if google else []
        out.append({"title": title, "link": link, "site": source or site_of(link), "direct": not google,
                    "published": _when(it.findtext("pubDate")), "summary": "" if google else _text(desc)[:400],
                    "cluster": cluster, "feed": name})
    return out


def read_trends(name, url):
    root = ET.fromstring(_get(url, timeout=20, tries=2))
    out = []
    for it in root.iter("item"):
        term = _text(it.findtext("title"))
        traffic = (it.findtext(HT + "approx_traffic") or "").replace(",", "").replace("+", "").strip().upper()
        mult = 1000 if traffic.endswith("K") else 1_000_000 if traffic.endswith("M") else 1
        try:
            n = int(float(traffic.rstrip("KM") or 0) * mult)
        except ValueError:
            n = 0
        titles = [_text(n_.findtext(HT + "news_item_title")) for n_ in it.findall(HT + "news_item")]
        out.append({"term": term, "traffic": n, "country": name, "titles": titles})
    return out


def collect(log=print):
    """All recent articles and trending searches. Feeds that fail are skipped (listed in problems)."""
    articles, trends, problems = [], [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(read_feed, n, u): n for n, u in FEEDS}
        jobs.update({pool.submit(read_trends, n, u): "Google Trends " + n for n, u in TRENDS})
        for f, n in jobs.items():
            try:
                res = f.result(timeout=60)
                (trends if n.startswith("Google Trends") else articles).extend(res)
            except Exception as e:
                problems.append(f"{n}: {e}")
    cutoff = time.time() - MAX_AGE_H * 3600
    seen, fresh = set(), []
    for a in articles:
        if a["link"] in seen or (a["published"] and a["published"] < cutoff):
            continue
        seen.add(a["link"]); fresh.append(a)
    log(f"read {len(fresh)} articles and {len(trends)} trending searches ({len(problems)} feed(s) failed)")
    return fresh, trends, problems


# ---------- grouping the same story ----------

STOP |= set("""one two three four five six seven eight nine ten dozen hundred thousand lakh crore million billion
january february march april may june july august september october november december jan feb mar apr jun jul aug sep sept
oct nov dec monday tuesday wednesday thursday friday saturday sunday morning evening night tonight yesterday tomorrow""".split())


def keywords(text):
    words = set()
    for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", (text or "").lower().replace("’", "'")):
        w = w.split("'")[0]
        if w in STOP or len(w) < 3 or w.isdigit():
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.add(w)
    return words


def similar(a, b):
    shared = len(a & b)
    return shared >= 3 or (shared >= 2 and shared / max(1, min(len(a), len(b))) >= 0.5)


def group(articles):
    """Same-story test weighted by how rare each word is in this scan: two headlines sharing names or places
    ("Kota", "pilgrims") match, while sharing common news words ("killed", "box office") does not.
    An article joins a story if it matches the story's first headline or at least half of its headlines (no chaining)."""
    for a in articles:
        a["kw"] = keywords(a["title"])
    df = {}
    for a in articles:
        for w in a["kw"]:
            df[w] = df.get(w, 0) + 1
    n = max(2, len(articles))
    idf = {w: math.log(n / c) for w, c in df.items()}

    def weight(ws):
        return sum(idf.get(w, 0) for w in ws)

    def same(a, b):
        shared = a & b
        if len(shared) < 2:
            return False
        w, wa, wb = weight(shared), weight(a), weight(b)
        return w >= 8 and w >= 0.4 * min(wa, wb) and w >= 0.3 * max(wa, wb)

    clusters = []
    for a in sorted(articles, key=lambda a: -(a["published"] or 0)):
        if len(a["kw"]) < 2:
            continue
        best = None
        for c in clusters:
            if same(a["kw"], c[0]["kw"]) or sum(same(a["kw"], m["kw"]) for m in c) * 2 >= len(c):
                best = c
                break
        (best.append(a) if best is not None else clusters.append([a]))
    return clusters


def core(members):
    """The words that describe the story: those used by at least two of its headlines (or all of a lone headline's)."""
    count = {}
    for m in members:
        for w in m["kw"]:
            count[w] = count.get(w, 0) + 1
    words = [w for w, n in count.items() if n >= 2] if len(members) > 1 else list(count)
    if len(words) < 2:
        words = list(count)
    return sorted(words, key=lambda w: -count[w])[:12]


# ---------- scoring ----------

def trend_match(story_kw, trends):
    best = None
    for t in trends:
        term_kw = keywords(t["term"])
        hit = (term_kw and term_kw <= story_kw) or similar(story_kw, term_kw | set().union(*[keywords(x) for x in t["titles"]] or [set()]))
        if hit and (best is None or t["traffic"] > best["traffic"]):
            best = t
    return best


def ollama_up():
    try:
        urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3).read()
        return True
    except Exception:
        return False


def start_ollama():
    """Ollama usually runs in the background already; if not, start it quietly."""
    if ollama_up():
        return True
    exe = shutil.which("ollama") or os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
    if not os.path.exists(exe):
        return False
    subprocess.Popen([exe, "serve"], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(20):
        time.sleep(1)
        if ollama_up():
            return True
    return False


AI_PROMPT = """You choose news stories for short Instagram news reels (30 to 45 seconds) for viewers in India who also follow world news.
Rate each story from 1 to 10 for how many people would watch and share a reel about it.
10 = huge, surprising or emotional: a disaster, a famous name, a shocking twist, big money, something that changes ordinary people's lives.
5 = solid news, but ordinary.
1 = dull, niche or routine: minor statements, local admin news, routine court hearings, live scores, share prices.
Lower the score if the story is hard to explain in 30 seconds.

Stories:
{stories}

Reply with JSON only: {{"scores": [{{"n": 1, "score": 7, "why": "2 to 5 words"}}]}} with one entry for every story.
"why" names the reason in 2 to 5 words (for example "famous name, big money"); never repeat the headline."""


def ai_rate(stories, log=print):
    """{story id: (score 1-10, why)} for the stories the AI could rate; the rest stay unrated (tried again next scan)."""
    if not stories or not start_ollama():
        if stories:
            log("the laptop AI (Ollama) is not running: reel potential not rated this time")
        return {}
    rated = {}
    for i in range(0, len(stories), AI_BATCH):
        part = stories[i:i + AI_BATCH]
        lines = "\n".join(f"{k}. {s['title']}" + (f" ({s['summary'][:160]})" if s.get("summary") else "")
                          for k, s in enumerate(part, 1))
        body = json.dumps({"model": AI_MODEL, "prompt": AI_PROMPT.format(stories=lines), "format": "json", "stream": False,
                           "options": {"temperature": 0.2, "num_ctx": 4096}}).encode()
        try:
            req = urllib.request.Request(OLLAMA + "/api/generate", data=body, headers={"Content-Type": "application/json"})
            reply = json.loads(json.loads(urllib.request.urlopen(req, timeout=300).read())["response"])
            items = reply.get("scores", reply if isinstance(reply, list) else [])
            for it in items:
                n, sc = int(it.get("n", 0)), float(it.get("score", 0))
                if 1 <= n <= len(part) and 1 <= sc <= 10:
                    rated[part[n - 1]["id"]] = (sc, str(it.get("why", ""))[:80])
        except Exception as e:
            log(f"the laptop AI could not rate {len(part)} stories: {e}")
    return rated


def pick_title(members, kw):
    def rank(m):  # a plain headline beats a "Live updates" page
        live = bool(re.search(r"\blive\b|\bupdates\b|horoscope", m["title"], re.I))
        return (not live, len(m["kw"] & kw), m["direct"], m["published"] or 0)
    return max(members, key=rank)


def build(articles, trends, old, decided, made_topics, now=None, log=print):
    """Turn articles into scored stories. old: {id: story} from earlier scans (keeps ids, first-seen times, AI ratings and
    history). decided: {id: {"core": [...]}} stories you made or skipped (never shown again). made_topics: reel topics."""
    now = now or time.time()
    old_list = list(old.values())
    decided_kw = [set(d.get("core", [])) for d in decided.values()]
    made_kw = [keywords(t) for t in made_topics]
    stories = []
    for members in group(articles):
        kw_list = core(members)
        kw = set(kw_list)
        if any(similar(kw, d) for d in decided_kw):
            continue  # you already made or skipped this story
        if any(similar(kw, m) for m in made_kw if len(m) >= 2):
            continue  # there's already a reel about it
        prev = next((o for o in old_list if similar(kw, set(o["core"]))), None)
        if prev:
            old_list.remove(prev)
        lead = pick_title(members, kw)
        sites = sorted({m["site"] for m in members} | {s for m in members for s in m["cluster"]})
        newest = max((m["published"] or now) for m in members)
        recent = {m["site"] for m in members if m["published"] and now - m["published"] <= RECENT_H * 3600}
        history = [h for h in (prev or {}).get("history", []) if now - h[0] <= RECENT_H * 3600] + [[now, len(sites)]]
        gained = len(sites) - history[0][1]
        trend = trend_match(kw | lead["kw"], trends)
        cov = 25 * min(1, math.log1p(len(sites)) / math.log1p(15))
        tr = 15 * min(1, math.log10(max(trend["traffic"], 10)) / 6) if trend else 0
        growth = min(20, max(5 * len(recent), 4 * gained))
        fresh = 10 * max(0, 1 - (now - newest) / (MAX_AGE_H * 3600))
        links = sorted(members, key=lambda m: (not m["direct"], -(m["published"] or 0)))
        summary = next((m["summary"] for m in links if m["summary"]), "")
        story = {"id": (prev or {}).get("id") or uuid.uuid4().hex[:10], "title": lead["title"], "summary": summary[:300],
                 "core": kw_list, "sites": sites[:40], "site_count": len(sites), "newest": newest,
                 "first_seen": (prev or {}).get("first_seen", now), "history": history,
                 "links": [{"site": m["site"], "url": m["link"], "title": m["title"]} for m in links[:8]],
                 "reel_links": [m["link"] for m in links if m["direct"]][:3],
                 "trend": {"term": trend["term"], "traffic": trend["traffic"], "country": trend["country"]} if trend else None,
                 "ai": (prev or {}).get("ai"),
                 "parts": {"popular": round(cov + tr, 1), "growth": round(growth, 1), "fresh": round(fresh, 1)}}
        stories.append(story)
    return stories


def finish_scores(stories, log=print):
    """Rate new stories with the laptop AI (most popular first), then add up each story's total and its reasons."""
    todo = sorted([s for s in stories if not s.get("ai")], key=lambda s: -(s["parts"]["popular"] + s["parts"]["growth"]))
    if todo:
        log(f"rating {min(len(todo), AI_MAX_NEW)} new stories with the laptop AI")
    for sid, (sc, why) in ai_rate(todo[:AI_MAX_NEW], log).items():
        next(s for s in stories if s["id"] == sid)["ai"] = {"score": sc, "why": why}
    for s in stories:
        p = s["parts"]
        p["potential"] = round(3 * s["ai"]["score"], 1) if s.get("ai") else 15.0  # unrated: middle value for now
        s["score"] = int(round(min(100, p["popular"] + p["growth"] + p["potential"] + p["fresh"])))
        why = [f"{s['site_count']} site{'s' if s['site_count'] != 1 else ''} covering it"]
        if s.get("trend"):
            t = s["trend"]
            why.append(f"trending on Google {t['country']}" + (f" ({t['traffic']:,}+ searches)" if t["traffic"] else ""))
        if p["growth"] >= 10:
            why.append("growing fast")
        if s.get("ai") and 0 < len(s["ai"]["why"]) <= 40:  # the small AI sometimes just repeats the headline
            why.append(s["ai"]["why"])
        s["reasons"] = why
    stories.sort(key=lambda s: -s["score"])
    return stories


def scan(old, decided, made_topics, log=print):
    articles, trends, problems = collect(log)
    if not articles:
        raise RuntimeError("no news could be read (is the internet on?) " + "; ".join(problems[:3]))
    stories = build(articles, trends, old, decided, made_topics, log=log)
    log(f"{len(stories)} stories after grouping")
    return finish_scores(stories, log), problems
