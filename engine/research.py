"""Research step (runs on your laptop): gather source text for a reel from
  - your PDFs, your links, and pasted text
  - optional free web search: Bing News RSS (recent news) + Wikipedia (background)
and trim it to the parts most relevant to the topic, so it fits in the LLM's prompt.
"""
import html, json, re, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ReelStudio/1.0"}
TOTAL_CHARS = 14000  # roughly 3,500 tokens of source text for the LLM


def _get(url, timeout=25):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def read_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else None) or str(path).split("\\")[-1]
    if len(text.strip()) < 200:
        raise ValueError("this PDF has almost no selectable text (it may be a scan); paste its text instead")
    return {"title": title, "url": "", "kind": "pdf", "text": text}


def read_link(url):
    import trafilatura
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError("could not open the page (it may block programs or need a login)")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""
    meta = trafilatura.extract_metadata(downloaded)
    if len(text) < 200:
        raise ValueError("found almost no article text on the page")
    return {"title": (meta.title if meta and meta.title else url), "url": url, "kind": "link", "text": text,
            "image": (meta.image if meta and meta.image and meta.image.startswith("http") else ""),
            "site": (meta.sitename if meta and meta.sitename else urllib.parse.urlparse(url).netloc)}


def search_news(topic, n=4):
    """Recent news articles from Bing News RSS (free, no key)."""
    q = urllib.parse.urlencode({"q": topic, "format": "rss"})
    root = ET.fromstring(_get("https://www.bing.com/news/search?" + q))
    urls = []
    for item in root.iter("item"):
        link = item.findtext("link") or ""
        real = urllib.parse.parse_qs(urllib.parse.urlparse(link).query).get("url", [link])[0]
        if real.startswith("http") and real not in urls:
            urls.append(real)
    found, errors = [], []
    for url in urls[: n * 3]:
        if len(found) >= n:
            break
        try:
            src = read_link(url)
            src["kind"] = "news"
            found.append(src)
        except Exception as e:
            errors.append(f"{url}: {e}")
    return found, errors


def search_wikipedia(topic):
    """Background from the best-matching Wikipedia article (free, no key)."""
    q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": topic, "format": "json", "srlimit": 1})
    hits = json.loads(_get("https://en.wikipedia.org/w/api.php?" + q))["query"]["search"]
    if not hits:
        return None
    title = hits[0]["title"]
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "titles": title,
                                "format": "json", "redirects": 1})
    pages = json.loads(_get("https://en.wikipedia.org/w/api.php?" + q))["query"]["pages"]
    text = next(iter(pages.values())).get("extract", "")
    if len(text) < 200:
        return None
    return {"title": f"Wikipedia: {title}", "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
            "kind": "wikipedia", "text": text, "wiki_title": title}


def wikipedia_images(topic, title=None, limit=12):
    """Photos used on the topic's Wikipedia article (all hosted on Wikimedia Commons, so freely licensed)."""
    if not title:
        q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": topic, "format": "json", "srlimit": 1})
        hits = json.loads(_get("https://en.wikipedia.org/w/api.php?" + q))["query"]["search"]
        if not hits:
            return []
        title = hits[0]["title"]
    q = urllib.parse.urlencode({"action": "query", "format": "json", "titles": title, "redirects": 1,
                                "generator": "images", "gimlimit": 50, "prop": "imageinfo",
                                "iiprop": "url|size|extmetadata", "iiurlwidth": 1280})
    pages = json.loads(_get("https://en.wikipedia.org/w/api.php?" + q)).get("query", {}).get("pages", {})
    out = []
    for p in pages.values():
        info = (p.get("imageinfo") or [{}])[0]
        name = p.get("title", "")
        if not re.search(r"\.(jpe?g|png|webp)$", name, re.I) or min(info.get("width", 0), info.get("height", 0)) < 600:
            continue  # skips icons, logos (usually .svg), maps and tiny images
        if re.search(r"logo|icon|flag|map|signature|symbol|seal|coat of arms", name, re.I):
            continue
        meta = info.get("extmetadata") or {}
        artist = re.sub("<[^>]+>", "", (meta.get("Artist") or {}).get("value", "unknown")).strip()
        lic = (meta.get("LicenseShortName") or {}).get("value", "free license")
        clean = name.removeprefix("File:").rsplit(".", 1)[0]
        out.append({"url": info.get("thumburl") or info.get("url"), "title": clean, "kind": "wikipedia",
                    "credit": f'"{clean}" by {artist}, {lic}, via Wikimedia Commons (from Wikipedia: {title})'})
    return out[:limit]


def photo_pool(topic, sources, press_photos=False, log=print):
    """Photos picked up during research, tried before any general photo search:
    the topic's Wikipedia article images (free) and, only if you opt in, the main image of each article/link
    (usually copyrighted news or press photos, credited to the site)."""
    pool, problems = [], []
    wiki_title = next((s.get("wiki_title") for s in sources if s.get("wiki_title")), None)
    try:
        imgs = wikipedia_images(topic, wiki_title)
        pool += imgs
        log(f"wikipedia photos: {len(imgs)}")
    except Exception as e:
        problems.append(f"Wikipedia photos failed: {e}")
    if press_photos:
        press = [{"url": s["image"], "title": s["title"], "kind": "press",
                  "credit": f"Image from {s.get('site') or s['url']} ({s['url']}) - may be copyrighted"}
                 for s in sources if s.get("image")]
        pool += press
        log(f"press photos from articles: {len(press)}")
    seen, unique = set(), []
    for p in pool:
        if p["url"] and p["url"] not in seen:
            seen.add(p["url"]); unique.append(p)
    return unique, problems


def _keywords(topic):
    stop = {"the", "and", "for", "with", "that", "this", "from", "about", "what", "when", "your", "into", "how", "why"}
    return [w for w in re.findall(r"[a-z0-9]+", topic.lower()) if len(w) > 2 and w not in stop]


def _trim(text, topic, budget):
    """Keep the paragraphs that mention the topic most, in their original order, within the budget."""
    text = html.unescape(re.sub(r"[ \t]+", " ", text))
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[A-Z])", text) if len(p.strip()) > 40]
    if sum(len(p) for p in paras) <= budget:
        return "\n".join(paras)
    keys = _keywords(topic)
    scored = sorted(range(len(paras)), key=lambda i: (-sum(paras[i].lower().count(k) for k in keys), i))
    keep, used = set(), 0
    for i in [0] + scored:  # always keep the opening paragraph
        if i in keep or used + len(paras[i]) > budget:
            continue
        keep.add(i)
        used += len(paras[i])
    return "\n".join(paras[i] for i in sorted(keep))


def gather(topic, links=(), pdf_paths=(), pasted="", web=False, log=print, wiki=False):
    """Returns (sources, problems). Each source: {id, title, url, kind, text}.
    web: news search + Wikipedia. wiki: Wikipedia only (grounds General topics in a real source)."""
    sources, problems = [], []
    if pasted and pasted.strip():
        sources.append({"title": "Text you pasted", "url": "", "kind": "pasted", "text": pasted})
    for p in pdf_paths:
        try:
            sources.append(read_pdf(p)); log(f"read PDF: {p}")
        except Exception as e:
            problems.append(f"PDF {p}: {e}")
    for url in links:
        url = url.strip()
        if not url:
            continue
        try:
            sources.append(read_link(url)); log(f"read link: {url}")
        except Exception as e:
            problems.append(f"Link {url}: {e}")
    if web:
        try:
            news, errs = search_news(topic)
            sources += news; problems += [f"News: {e}" for e in errs[:3]]
            log(f"web search: {len(news)} news articles")
        except Exception as e:
            problems.append(f"News search failed: {e}")
    if web or wiki:
        try:
            article = search_wikipedia(topic)
            if article:
                sources.append(article); log(f"wikipedia: {article['title']}")
        except Exception as e:
            problems.append(f"Wikipedia search failed: {e}")
    if sources:
        mine = [s for s in sources if s["kind"] in ("pasted", "pdf", "link")]
        # your own sources get twice the space of web results
        weights = [2 if s in mine else 1 for s in sources]
        unit = TOTAL_CHARS / sum(weights)
        for k, (s, w) in enumerate(zip(sources, weights), 1):
            s["id"] = f"S{k}"
            s["text"] = _trim(s["text"], topic, int(unit * w))
    return sources, problems


def as_prompt_text(sources):
    return "\n\n".join(f"[{s['id']}] {s['title']}\n{s['text']}" for s in sources)


if __name__ == "__main__":  # quick manual test: python research.py "topic"
    import sys
    srcs, probs = gather(sys.argv[1], web=True)
    for s in srcs:
        print(s["id"], s["kind"], s["title"], s["url"], len(s["text"]))
    print("problems:", probs)
