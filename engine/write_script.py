"""Write a clear, fun reel script with a local LLM (Ollama). Clear first, fun second.
  1. brainstorm 8 hooks that anyone understands instantly
  2. an editor picks the hook that is clearest AND most curiosity-making
  3. write the script as a story in order, around that hook (tone from examples.json)
  4. "12-year-old check": rewrite only lines that are confusing, jumpy or slangy (hook stays locked)
With --sources, every fact must come from the given source text, and each line is tagged with its source.

Usage:  python write_script.py "TOPIC" OUT.json [--sources sources.json] [--only-sources] [--news]
                               [--subject "octopus|polpo"] [--model gemma3:27b]
"""
import argparse, json, re, time, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
EXAMPLES = json.loads((HERE / "scripts" / "examples.json").read_text(encoding="utf-8-sig"))
SLANG = re.compile(r"\b(vibes?|fancy|low-?key|slay|no cap|drama queens?|literally|bestie|bruh|vibe check|iconic)\b", re.I)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
MODEL = "gemma3:27b"

VOICE = """You write short learning videos (Reels / TikTok) that anyone can follow on the first listen.
Clear first, fun second.
Rules:
- Each line is ONE simple, complete sentence (or two very short ones) with ONE idea. 8-18 words per line.
- Use everyday words a 12-year-old knows. If you need a technical word, say what it means in plain words.
- Tell it as a story, in order: hook -> what it is -> how it works -> the surprising part -> why it happens
  -> what that means -> one bonus fact -> call to action.
- Every line must follow from the line before it. No jumps, no riddles, no repeating a fact.
- No slang ("vibe", "fancy", "lowkey", "slay", "no cap", "drama queen", "literally"), no emojis, no hashtags.
- Only use a comparison if it is instantly clear, and say what it compares.
- Talk to the viewer as "you". The fun comes from the surprising facts, not from jokes. At most one light joke.
- Every fact must be true. No made-up numbers."""

NEWS_RULES = """This is a NEWS reel. Extra rules:
- Report what happened, who said what, and when. Stay neutral: do not take sides or give opinions.
- If sources disagree, say so. Attribute claims ("the company says...", "police said...").
- If a source says "reportedly", "plans to", "could" or "may", keep that word: never turn a report into a fact.
- Use exact numbers and names only as they appear in the sources."""
CTA = re.compile(r"\b(follow|comment|save|share|subscribe|send this|tag)\b", re.I)


def source_rules(only):
    if only:
        return ("Use ONLY facts stated in the SOURCE MATERIAL below. If a fact is not in the sources, leave it out, "
                "even if you think you know it.")
    return ("Base the script on the SOURCE MATERIAL below. You may add only very well-known background facts. "
            "Never contradict the sources.")


def examples_text():
    return "\n\n".join(f"On-screen hook: {ex['hook']!r}\n" + "\n".join(f"- {l}" for l in ex["lines"]) for ex in EXAMPLES)


def ask(model, prompt, temperature, ctx=12288):
    body = json.dumps({"model": model, "prompt": prompt, "format": "json", "stream": False,
                       "options": {"temperature": temperature, "num_ctx": ctx}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"})
    return json.loads(json.loads(urllib.request.urlopen(req, timeout=900).read())["response"])


def step(name, fn):
    t = time.perf_counter()
    for attempt in range(1, 4):
        try:
            result = fn()
            print(f"[{name}] done in {time.perf_counter() - t:.1f}s", flush=True)
            return result
        except Exception as e:  # any odd reply shape: retry
            print(f"[{name}] attempt {attempt} unusable ({e}), retrying", flush=True)
    raise SystemExit(f"{name} failed 3 times")


def field(d, names):
    """Value for the first matching key (exact, then partial match, case-insensitive); '' if none."""
    if not isinstance(d, dict):
        return ""
    low = {str(k).lower(): v for k, v in d.items()}
    for n in names:
        if n in low and low[n] not in (None, ""):
            return str(low[n]).strip()
    for n in names:
        for k, v in low.items():
            if n in k and isinstance(v, (str, int, float)) and str(v).strip():
                return str(v).strip()
    return ""


def first_list(data, preferred=None):
    """The list in a model reply: data itself, data[preferred], or the first list value."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if preferred and isinstance(data.get(preferred), list):
            return data[preferred]
        for v in data.values():
            if isinstance(v, list):
                return v
    raise ValueError("no list in reply")


def clean(text):
    return re.sub(r"\s+", " ", EMOJI.sub("", str(text)).replace("*", "")).strip()


def clean_screen(text):
    """Like clean(), but keeps the headline's line break (real newline or a literal backslash-n)."""
    return "\n".join(filter(None, (clean(part) for part in str(text or "").replace("\\n", "\n").split("\n"))))


def scene_parts(s, subject):
    """[line, photo, source] from whatever shape the model used."""
    if isinstance(s, str):
        line, photo, src = s, "", ""
    elif isinstance(s, (list, tuple)):
        line, photo, src = (list(s) + ["", "", ""])[:3]
    else:
        line = field(s, ("line", "narration", "text", "spoken", "voice"))
        photo = next((v for k, v in s.items() if any(w in k.lower() for w in ("photo", "image", "search", "visual"))), "")
        src = field(s, ("source", "src", "ref"))
    line = clean(line)
    if not line:
        raise ValueError(f"scene without a line: {s}")
    return [line, str(photo).strip() or subject, str(src).strip().upper().strip("[]")]


def problems(lines):
    """Lines that break the clarity rules in ways code can check."""
    out = []
    seen = set()
    for k, l in enumerate(lines):
        words = len(re.findall(r"\w+", l))
        if SLANG.search(l):
            out.append(f"line {k + 1} uses slang")
        if words > 26:
            out.append(f"line {k + 1} is too long ({words} words)")
        key = " ".join(sorted(set(re.findall(r"[a-z]{4,}", l.lower()))))
        if key and key in seen:
            out.append(f"line {k + 1} repeats an earlier line")
        seen.add(key)
    if lines and not CTA.search(lines[-1]):
        out.append(f"line {len(lines)} must be a call to action (follow, comment, save or share)")
    return out


N_HOOKS = 15
HOOK_EXAMPLES = """- "This animal's heart stops every time it swims."
- "Your banana is slightly radioactive."
- "Sharks are older than trees."
- "A day on Venus is longer than its whole year."
- "Cleopatra lived closer to the Moon landing than to the pyramids being built."
- "Why do octopuses hate swimming?\""""
FUNCTION_WORDS = set("""a an the is are was were be it its it's this that these those you your can can't do does did
not no why how what when who where will would has have had than then more most every all of in on at to for from
with by and or but if so as up down out over after before still just only never always really very ever""".split())


FILLER = re.compile(r"what the|seriously|whoa|wow|omg|no brainer|that's (wild|rough|crazy)|so (smart|cool)|mind.?blown"
                    r"|it's (tricky|a gland)|really\?|\.\.\.|…", re.I)
CONTEXT_START = re.compile(r"^(that|this|these|so|and|but|which|it|they)\b", re.I)


def natural_screen(screen, spoken=""):
    """True if the on-screen headline reads like a phrase, not a pile of keywords ("OCTOPUS TRICK HEART CONTROL"),
    has no reaction filler ("WHAT THE...", "NO BRAINER"), and (given the spoken hook) is about the same thing."""
    words = re.findall(r"[a-z']+", (screen or "").lower())
    if len(words) < 3 or len(screen) > 48 or FILLER.search(screen):
        return False
    if spoken:
        said = {w[:4] for w in re.findall(r"[a-z']+", spoken.lower()) if w not in FUNCTION_WORDS and len(w) > 2}
        if not any(w[:4] in said for w in words if w not in FUNCTION_WORDS and len(w) > 2):
            return False
    return "?" in screen or any(w in FUNCTION_WORDS for w in words)


def stands_alone(spoken):
    """A hook is the very first line, so it can't lean on a line before it ("That means...", "This is why...")."""
    return not CONTEXT_START.match(spoken.strip())


def wrap_screen(text, width=20):
    """Deterministic fallback: the spoken hook itself, upper-cased and wrapped onto at most 2 lines."""
    words, lines = text.upper().rstrip(".").split(), [""]
    for w in words:
        if len(lines[-1]) + len(w) + 1 > width and lines[-1]:
            if len(lines) == 2:
                lines[-1] = lines[-1].rstrip() + "..."
                break
            lines.append("")
        lines[-1] = (lines[-1] + " " + w).strip()
    return "\n".join(lines)


def make_screen(model, spoken, log=print):
    """Write a natural on-screen headline for a spoken hook (the code checks it, with a safe fallback)."""
    for _ in range(2):
        try:
            data = ask(model, f"""Write the on-screen headline for this reel hook: "{spoken}"
Rules: ALL CAPS, 2 short lines separated by \\n, max 20 characters per line, a natural phrase or question that
means the same as the hook. Never a string of keywords.
Good: "ITS HEART STOPS\\nWHEN IT SWIMS", "SHARKS ARE OLDER\\nTHAN TREES". Bad: "OCTOPUS TRICK\\nHEART CONTROL",
"THREE HEARTS?\\nWHAT THE...", "ARMS THINKING?\\nNO BRAINER". Use words from the hook itself.
Return JSON: {{"screen": "..."}}""", 0.4)
            screen = clean_screen(field(data, ("screen", "headline", "text"))).upper()
            if natural_screen(screen, spoken):
                return screen
        except Exception as e:
            log(f"  headline attempt failed ({e})")
    return wrap_screen(spoken)


def write(topic, out_path, sources=(), only_sources=False, news=False, subject_words="", model=MODEL, log=print,
          instructions="", hook_idea="", forced_hook=None, forced_options=None):
    t0 = time.perf_counter()
    src_text = "\n\n".join(f"[{s['id']}] {s['title']}\n{s['text']}" for s in sources)
    src_block = f"\n{source_rules(only_sources)}\n\nSOURCE MATERIAL:\n{src_text}\n" if sources else ""
    rules = VOICE + ("\n\n" + NEWS_RULES if news else "")
    instructions, hook_idea = clean(instructions)[:1500], clean(hook_idea)[:300]
    if instructions:  # the creator's own wishes, applied at every step
        rules += ("\n\nINSTRUCTIONS FROM THE CREATOR (follow these closely, unless they ask for something untrue"
                  f"{' or not supported by the sources' if sources else ''}):\n{instructions}")
    hook_block = (f"\nThe creator suggested this hook idea: \"{hook_idea}\"\nInclude one hook that is their idea, "
                  "cleaned up to fit the rules, and make at least 3 more hooks that are variations of it.\n"
                  if hook_idea else "")

    if forced_hook:  # "remake with this hook": the creator already chose it
        hook = {"spoken": clean(forced_hook["spoken"]), "screen": forced_hook.get("screen") or ""}
        if not natural_screen(hook["screen"].upper(), hook["spoken"]):
            hook["screen"] = make_screen(model, hook["spoken"], log)
        hook_options = list(forced_options or [])
        log(f"  using the chosen hook: {hook['spoken']}  |  {hook['screen']!r}")
    else:
        # 1. brainstorm 15 hooks that make sense instantly
        def hooks():
            raw = ask(model, f"""{rules}
{src_block}
Topic: {topic}

Write {N_HOOKS} different hooks for the first 2 seconds of this reel. Each has:
- "spoken": the first line the voice says. One plain sentence, max 12 words.
- "screen": the on-screen headline. A short natural phrase or question in ALL CAPS, 2 lines separated by \\n,
  max 20 characters per line. It must read like something a person would say, never a string of keywords.
  Good: "ITS HEART STOPS\\nWHEN IT SWIMS", "SHARKS ARE OLDER\\nTHAN TREES", "WHY DOES THIS\\nCAR FLY?"
  Bad: "OCTOPUS TRICK\\nHEART CONTROL", "SHARK SURVIVAL\\nMILLIONS YEARS", "OCTOPUS FACTS", and reaction filler like "WHAT THE...", "SERIOUSLY?!", "NO BRAINER", "WHOA".
A good hook is a clear, surprising, TRUE claim or question about the topic that anyone understands instantly,
and that makes them want the explanation. Good examples:
{HOOK_EXAMPLES}
Bad: riddles, metaphors that need explaining, vague teasers, labels, exaggerations ("on demand", "can control"),
and generic lines like "AI is changing quickly" or "Here's what you need to know".
Vary the angle: surprising fact, "you" question, myth vs truth, comparison, number, mini-story.
{"Each hook must lead with ONE specific, surprising fact from the sources: a number, a name or a result." if sources else ""}
{hook_block}
Return JSON: {{"hooks": [{{"spoken": "...", "screen": "..."}}]}}""", 0.9)
            data = []
            for h in first_list(raw):
                if isinstance(h, dict):
                    spoken = clean(field(h, ("spoken", "line", "hook", "voice", "text")))
                    if spoken and stands_alone(spoken) and spoken not in [d["spoken"] for d in data]:
                        screen = field(h, ("screen", "on_screen", "onscreen", "headline", "overlay", "caption"))
                        data.append({"spoken": spoken, "screen": clean_screen(screen)})
            if len(data) < 6:
                raise ValueError(f"only {len(data)} usable hooks")
            return data
        hook_list = step("1 hooks", hooks)
        for k, h in enumerate(hook_list):
            log(f"  {k}. {h['spoken']}  |  {h['screen']!r}" + ("" if natural_screen(h["screen"], h["spoken"]) else "  [screen text not natural]"))

        # 2. rank: clear AND curious AND true; keep the top 3 so the creator can remake with another one
        def rank():
            listing = "\n".join(f"{k}. {h['spoken']}" for k, h in enumerate(hook_list))
            data = ask(model, f"""You are the editor of a learning channel. Rank the best opening lines.
Each must pass ALL tests:
1. Clear: a 12-year-old understands it instantly on the first listen, with no explanation.
2. Curious: they really want to know why or how.
3. True: no exaggeration or misleading wording.
Reject lines that lean on a line before them ("That means...", "This is why..."), riddles, metaphors, vague teasers, generic lines, and lines that give the whole answer away.
Prefer a hook with a specific number, name or result.
{"It must also be accurate to the sources and neutral." if news else ""}
{f'The creator suggested this hook idea: "{hook_idea}". If a hook based on it passes the tests, rank it first.' if hook_idea else ""}
{f"The creator's instructions for this reel: {instructions}" if instructions else ""}

{listing}

Return JSON with the 3 best, best first: {{"top": [{{"index": <number>, "why": "<one short sentence>"}}]}}""", 0.2)
            raw = data.get("top", data.get("ranking", [])) if isinstance(data, dict) else data
            picks = []
            for r in raw if isinstance(raw, list) else []:
                try:
                    i = int(str(field(r, ("index", "number", "best")) if isinstance(r, dict) else r).strip(". "))
                except ValueError:
                    continue
                if 0 <= i < len(hook_list) and i not in [p[0] for p in picks]:
                    picks.append((i, field(r, ("why", "reason")) if isinstance(r, dict) else ""))
            if not picks:
                raise ValueError("no usable ranking")
            return picks[:3]
        picks = step("2 rank", rank)
        hook_options = []
        for i, why in picks:
            h = dict(hook_list[i], why=clean(why))
            h["screen"] = h["screen"].upper()
            if not natural_screen(h["screen"], h["spoken"]):  # keyword soup or filler: write a proper headline
                h["screen"] = make_screen(model, h["spoken"], log)
            hook_options.append(h)
        hook = hook_options[0]
        for n, h in enumerate(hook_options, 1):
            log(f"  top {n}: {h['spoken']}  |  {h['screen']!r}  ({h.get('why', '')})")

    # 3. write the story
    def write_story():
        data = ask(model, f"""{rules}

Example scripts with the clarity and tone we want (different topics; copy the style, not the words):

{examples_text()}
{src_block}
Now write a new script.
Topic: {topic}
Line 1 must be exactly: {hook['spoken']}

7 lines in total, 70-110 spoken words. Follow the story order from the rules.
Line 7 must be a short call to action (follow, comment, save or share) and nothing else.
Choose "photo_subject": ONE simple, common English noun for the photos (for example "octopus", "robot",
"computer", "city", "protest", "rocket"){f' - use "{subject_words.split("|")[0]}"' if subject_words else ""}.
For each line give "photo": 2-4 plain words describing a real photo a free photo library would have
(always include the photo_subject word; no logos, no text, no diagrams)
{', and "source": the source id (like "S1") the fact comes from, or "" for the hook and call to action' if sources else ""}.
Return JSON: {{"photo_subject": "...", "scenes": [{{"line": "...", "photo": "..."{', "source": "S1"' if sources else ""}}}]}}""", 0.7)
        subject = (subject_words.split("|")[0] if subject_words else field(data, ("photo_subject", "subject")) or "nature")
        subject = re.sub(r"[^a-z ]", "", subject.lower()).strip() or "nature"
        scenes = [scene_parts(s, subject) for s in first_list(data, "scenes")]
        if not 5 <= len(scenes) <= 9:
            raise ValueError(f"{len(scenes)} scenes")
        scenes[0][0] = hook["spoken"]  # hook stays locked
        return subject, scenes
    subject, scenes = step("3 write", write_story)

    # 4. 12-year-old check: rewrite only unclear lines, same number of lines, hook locked
    for round_ in (1, 2):
        def check():
            draft = "\n".join(f"{k + 1}. {line}" for k, (line, _, _) in enumerate(scenes))
            issues = problems([s[0] for s in scenes])
            data = ask(model, f"""{rules}
{src_block}
Read this reel script as a 12-year-old hearing it once, with no pictures.
For each line ask: Is it one clear idea? Does it follow from the line before? Is every word easy?
Is it free of slang? Does it repeat an earlier line? {"Is it supported by the sources? Is it neutral?" if sources else ""}
Rewrite ONLY the lines that fail. Keep the facts. Keep line 1 exactly as it is.
Keep exactly {len(scenes)} lines, in the same order.
{"Known problems to fix: " + "; ".join(issues) if issues else ""}

Script:
{draft}

Return JSON: {{"lines": ["...", "..."]}}""", 0.3)
            lines = [scene_parts(l, subject)[0] for l in first_list(data, "lines")]
            if len(lines) != len(scenes):
                raise ValueError(f"got {len(lines)} lines, expected {len(scenes)}")
            return lines
        try:
            lines = step(f"4 clarity check #{round_}", check)
        except SystemExit:
            log("  clarity check failed, keeping the draft")
            break
        lines[0] = hook["spoken"]
        scenes = [[line, photo, src] for line, (_, photo, src) in zip(lines, scenes)]
        left = problems(lines)
        if not left:
            break
        log(f"  still to fix: {left}")

    # 5. fact check: fix wrong numbers, dates, order of events and cause/effect, change nothing else
    def fact_check():
        draft = "\n".join(f"{k + 1}. {line}" for k, (line, _, _) in enumerate(scenes))
        data = ask(model, f"""You are a careful fact-checker for an educational video.
{src_block}
Topic: {topic}
Check every line of this script for factual errors: wrong numbers or dates, wrong order of events
(for example "X appeared 385 million years AFTER Y" when it was 385 million years AGO), wrong cause and effect,
or claims {"not supported by the source material" if sources else "that are false or very doubtful"}.
Check line 1 (the hook) just as strictly: it must not exaggerate or mislead (for example "controls its heart on
demand" when the heart simply stops while swimming). If it does, fix it so it stays short, surprising and true.
For each line with a problem, give a corrected line in the same simple style and about the same length.
Do not touch lines that are correct. Do not rewrite for style.

Script:
{draft}

Return JSON: {{"fixes": [{{"line": <number>, "problem": "<short>", "fixed": "<corrected line>"}}]}}
Return {{"fixes": []}} if every line is correct.""", 0.1)
        fixes = []
        raw = data.get("fixes", []) if isinstance(data, dict) else data
        for f in raw if isinstance(raw, list) else []:
            if not isinstance(f, dict):
                continue
            try:
                n = int(str(field(f, ("line", "number", "index"))).strip(". "))
            except ValueError:
                continue
            fixed = clean(field(f, ("fixed", "correction", "corrected", "new")))
            if 1 <= n <= len(scenes) and fixed and fixed != scenes[n - 1][0]:
                fixes.append((n, field(f, ("problem", "issue", "error")), fixed))
        return fixes
    try:
        for n, problem, fixed in step("5 fact check", fact_check):
            log(f"  fact fix, line {n}: {problem}\n    was: {scenes[n - 1][0]}\n    now: {fixed}")
            scenes[n - 1][0] = fixed
    except SystemExit:
        log("  fact check failed, keeping the script as it is")

    if scenes[0][0] != hook["spoken"]:  # the fact check corrected the hook: the headline must match the true line
        log(f"  hook corrected, rewriting the on-screen headline (was {hook['screen']!r})")
        hook = dict(hook, spoken=scenes[0][0], screen=make_screen(model, scenes[0][0], log))
        if hook_options and not forced_hook:  # so "remake with this hook" uses the corrected version too
            hook_options[0].update(spoken=hook["spoken"], screen=hook["screen"])
    screen = "\n".join(l.strip().upper() for l in hook["screen"].split("\n")[:2])
    if not natural_screen(screen, hook["spoken"]):
        screen = wrap_screen(hook["spoken"])

    must = subject_words or subject
    ids = {s["id"] for s in sources}
    script = {
        "topic": topic, "style": "news" if news else "clear story", "hook": screen, "hook_spoken": hook["spoken"],
        "hook_options": [{"spoken": h["spoken"], "screen": h["screen"], "why": h.get("why", "")} for h in hook_options],
        "scenes": [[line, photo if subject in photo.lower() else f"{subject} {photo}", must] for line, photo, _ in scenes],
        "line_sources": [src if src in ids else "" for _, _, src in scenes],
        "sources": [{"id": s["id"], "title": s["title"], "url": s.get("url", "")} for s in sources],
    }
    Path(out_path).write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    words = sum(len(re.findall(r"\w+", s[0])) for s in script["scenes"])
    log("\nFINAL SCRIPT")
    log("hook on screen: " + screen.replace("\n", " / "))
    for k, (s, src) in enumerate(zip(script["scenes"], script["line_sources"]), 1):
        log(f"{k}. {s[0]}   [photo: {s[1]}]{f'  [{src}]' if src else ''}")
    log(f"script written in {time.perf_counter() - t0:.1f}s ({words} spoken words)")
    return script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic")
    ap.add_argument("out")
    ap.add_argument("--sources", help="JSON list of {id, title, url, text} from research.py")
    ap.add_argument("--only-sources", action="store_true")
    ap.add_argument("--news", action="store_true")
    ap.add_argument("--subject", default="", help='photo subject + translations, e.g. "octopus|polpo" (optional)')
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--instructions", default="", help="what to focus on, audience, tone, things to avoid")
    ap.add_argument("--hook-idea", default="", help="a hook you'd like the writer to use or build on")
    ap.add_argument("--forced-hook", default="", help='JSON {"spoken","screen"}: use this hook, skip brainstorming')
    args = ap.parse_args()
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8")) if args.sources else []
    write(args.topic, args.out, sources, args.only_sources, args.news, args.subject, args.model,
          instructions=args.instructions, hook_idea=args.hook_idea,
          forced_hook=json.loads(args.forced_hook) if args.forced_hook else None)


if __name__ == "__main__":
    main()
