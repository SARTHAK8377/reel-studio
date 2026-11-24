"""Reel Studio: type a topic (optionally add PDFs, links or text), and a reel is made in the background on Kaggle.

Run:  .venv\\Scripts\\python.exe app.py      then open http://127.0.0.1:5050
"""
import base64, json, os, queue, re, shutil, subprocess, sys, threading, time, traceback, uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory

HERE = Path(__file__).parent
ENGINE = HERE / "engine"
JOBS = HERE / "jobs"
KAGGLE = str(Path(sys.executable).parent / "kaggle.exe")
SLUG = "reel-studio-worker"
sys.path.insert(0, str(ENGINE))
import research  # noqa: E402
import discover  # noqa: E402
import instagram  # noqa: E402

app = Flask(__name__)
work_queue = queue.Queue()
lock = threading.Lock()


# ---------- job storage ----------

def job_path(jid):
    return JOBS / jid / "job.json"


def load(jid):
    return json.loads(job_path(jid).read_text(encoding="utf-8"))


STOP_FIELDS = ("stopped", "status", "stage", "error")


def save(job):
    with lock:
        try:  # the Stop button wrote to disk while the worker held an older copy: keep the stop
            disk = json.loads(job_path(job["id"]).read_text(encoding="utf-8"))
            if disk.get("stopped") and not job.get("stopped"):
                job.update({k: disk[k] for k in STOP_FIELDS if k in disk})
        except (OSError, ValueError):
            pass
        job_path(job["id"]).write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")


def update(job, **changes):
    job.update(changes)
    save(job)


def note(job, text):
    job.setdefault("log", []).append(f"{datetime.now():%H:%M:%S}  {text}")
    save(job)


# ---------- Kaggle ----------

NETWORK_TROUBLE = re.compile(r"HTTPSConnectionPool|SSLError|ConnectTimeout|ConnectionError|Read timed out|timed out|"
                             r"Temporary failure|Max retries exceeded", re.I)


KAGGLE_CREDS = Path.home() / ".kaggle" / "credentials.json"
_creds_lock = threading.Lock()


def renew_kaggle_login():
    """The Kaggle sign-in (kaggle auth login) lasts 12 hours. Kaggle's library only renews it 30 minutes AFTER it has
    expired (a flipped check in kagglesdk), so for half an hour every call is refused ("Authentication required").
    When the sign-in expires within 35 minutes, mark it as long expired so the library renews it on the next call.
    Only the expiry time is changed; the tokens are untouched."""
    from datetime import timedelta, timezone
    with _creds_lock:
        try:
            data = json.loads(KAGGLE_CREDS.read_text(encoding="utf-8"))
            expires = datetime.fromisoformat(data["access_token_expiration"])
        except (OSError, ValueError, KeyError, TypeError):
            return  # no OAuth sign-in file (an API token is used instead), nothing to do
        now = datetime.now(timezone.utc)
        if data.get("refresh_token") and now - timedelta(minutes=31) < expires < now + timedelta(minutes=35):
            data["access_token_expiration"] = (now - timedelta(minutes=31)).isoformat()
            tmp = KAGGLE_CREDS.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, KAGGLE_CREDS)


def kaggle(*args, timeout=300):
    """Run the Kaggle tool. A call that hangs (a bad internet moment) returns an error instead of crashing the run."""
    renew_kaggle_login()
    try:
        # the tool prints UTF-8 (set below); read it as UTF-8 too, or an unusual character in its output breaks the read
        r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except subprocess.TimeoutExpired:
        return 124, f"Kaggle did not answer within {timeout} s (internet trouble?)"
    return r.returncode, ((r.stdout or "") + (r.stderr or "")).strip()


_user = None


def kaggle_user():
    """Your Kaggle username (needed for the worker's id). None if not signed in yet."""
    global _user
    if _user:
        return _user
    code, out = kaggle("config", "view", timeout=60)
    m = re.search(r"username:\s*(\S+)", out)
    if m and m[1].lower() != "none":
        _user = m[1]
        return _user
    code, out = kaggle("kernels", "list", "--mine", "--page-size", "1", "--csv", timeout=60)
    m = re.search(r"^([\w-]+)/[\w-]+", out, re.M)
    if code == 0 and m:
        _user = m[1]
    return _user


def kaggle_status(ref):
    code, out = kaggle("kernels", "status", ref, timeout=60)
    if NETWORK_TROUBLE.search(out) and "KernelWorkerStatus" not in out:
        return "unknown: no answer from Kaggle (internet trouble), checking again soon"
    low = out.lower()
    for word in ("queued", "running", "complete", "error", "cancel"):
        if word in low:
            return word
    return "unknown: " + out[-200:]


CONFIG = HERE / "config.json"  # optional settings, e.g. {"flickr_api_key": "..."}


def settings():
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


MAX_BATCH = 5  # reels per Kaggle run; they share one setup (tools + LLM download)
MUSIC_MOODS = ("off", "upbeat", "calm", "curious", "dramatic", "tech")


def b64(obj):
    return base64.b64encode(json.dumps(obj).encode()).decode()


GPU_WAIT_MINUTES = None  # off: always wait for T4 x2 (the full gemma3:27b model). Set a number, e.g. 15, to switch to the P100 after that many minutes in the queue
GPUS = {"NvidiaTeslaT4": "T4 x2", "NvidiaTeslaP100": "P100"}


def build_worker(payloads, folder, machine="NvidiaTeslaT4"):
    files = {name: (ENGINE / name).read_text(encoding="utf-8")
             for name in ("make_reel_fast.py", "write_script.py", "scripts/examples.json")}
    cfg = settings()
    keys = {"flickr": cfg.get("flickr_api_key", ""), "pexels": cfg.get("pexels_api_key", ""),
            "pixabay": cfg.get("pixabay_api_key", "")}
    code = (ENGINE / "kaggle_worker_template.py").read_text(encoding="utf-8")
    code = code.replace("__JOBS__", b64(payloads)).replace("__KEYS__", b64(keys)).replace("__FILES__", b64(files))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "reel_worker.py").write_text(code, encoding="utf-8")
    meta = {"id": f"{kaggle_user()}/{SLUG}", "title": "Reel Studio Worker", "code_file": "reel_worker.py",
            "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
            "enable_tpu": "false", "enable_internet": "true", "machine_shape": machine,
            "dataset_sources": [], "competition_sources": [], "model_sources": [],
            # the one-time model cache run, if set up: saves downloading the 17 GB model every run
            "kernel_sources": [cfg["model_cache_kernel"]] if cfg.get("model_cache_kernel") else []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ---------- the background worker ----------

def fail(job, e):
    note(job, traceback.format_exc()[-800:] if isinstance(e, Exception) else str(e))
    update(job, status="error", stage="Failed", error=str(e))


def hooks_payload(job):
    """"Write new hooks" for a finished reel: reuse its research (no new web searches), avoid the hooks already shown."""
    update(job, stage="Getting ready to write new hooks", status="working")
    pdir = JOBS / job["parent_id"]
    parent = load(job["parent_id"])
    sources_file = pdir / "sources.json"
    sources = json.loads(sources_file.read_text(encoding="utf-8")) if sources_file.exists() else []
    shown = [h["spoken"] for h in (parent.get("script") or {}).get("hook_options", [])]
    return {"id": job["id"], "mode": "hooks", "topic": job["topic"], "news": job["news"],
            "only_sources": job["only_sources"], "sources": sources, "instructions": job.get("instructions", ""),
            "hook_idea": job.get("hook_idea", ""), "avoid": shown, "photo_pool": []}


def research_job(job):
    """Research on this laptop. Returns the payload the Kaggle worker needs for this reel."""
    if job.get("kind") == "hooks":
        return hooks_payload(job)
    jdir = JOBS / job["id"]
    update(job, stage="Researching", status="working")
    pdfs = sorted((jdir / "uploads").glob("*.pdf"))
    # every reel searches recent news, unless you asked for only your own sources
    sources, problems = research.gather(job["topic"], job["links"], pdfs, job["pasted"],
                                        web=job["news"] and not job["only_sources"],
                                        log=lambda m: note(job, m), wiki=not job["only_sources"])
    for p in problems:
        note(job, "skipped: " + p)
    if job["only_sources"] and not sources:
        raise RuntimeError("You chose 'only my sources', but none of them could be read.")
    (jdir / "sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
    note(job, f"{len(sources)} source(s) ready")
    photo_articles = []
    if job["only_sources"] and job.get("press_photos"):
        # facts stay limited to your own sources, but the story's photos can come from other news articles too
        try:
            photo_articles, _ = research.search_news(job["topic"])
            note(job, f"news articles read for photos only: {len(photo_articles)}")
        except Exception as e:
            note(job, f"skipped: news photo search failed: {e}")
    pool, problems = research.photo_pool(job["topic"], sources, job.get("press_photos", False), log=lambda m: note(job, m),
                                         photo_articles=photo_articles)
    for p in problems:
        note(job, "skipped: " + p)
    cfg = settings()
    off = [n for n, k in (("Flickr", "flickr_api_key"), ("Pexels", "pexels_api_key"), ("Pixabay", "pixabay_api_key"))
           if not cfg.get(k)]
    if off:
        note(job, f"off (no key in config.json): {', '.join(off)}")
    return {"id": job["id"], "topic": job["topic"], "news": job["news"], "only_sources": job["only_sources"],
            "sources": sources, "photo_pool": pool, "instructions": job.get("instructions", ""),
            "hook_idea": job.get("hook_idea", ""), "format": job.get("format", "9:16"),
            "voice": job.get("voice", ""), "music": job.get("music", "off"),
            "forced_hook": job.get("forced_hook"), "hook_options": job.get("hook_options")}


def run_batch(jobs):
    """Research each reel, then make them all in ONE Kaggle run (setup paid once), then hand out the results."""
    user = kaggle_user()
    payloads, ready = [], []
    for job in jobs:
        if job.get("stopped"):
            continue
        try:
            if not user:
                raise RuntimeError("Not signed in to Kaggle yet. Run the Kaggle sign-in once (see the top of the page).")
            payload = research_job(job)
            if not job.get("stopped"):  # stopped while researching
                payloads.append(payload); ready.append(job)
        except Exception as e:
            fail(job, e)
    if not ready:
        return
    folder = JOBS / ready[0]["id"] / "kaggle"
    for job in ready:
        update(job, stage="Sending to Kaggle", batch=[j["id"] for j in ready])
    ref = f"{user}/{SLUG}"

    def push(machine):
        build_worker(payloads, folder, machine)
        for attempt in range(1, 4):  # a connection error while sending: wait and send again
            code, out = kaggle("kernels", "push", "-p", str(folder), "--accelerator", machine)
            if "successfully pushed" in out or not NETWORK_TROUBLE.search(out) or attempt == 3:
                break
            for job in ready:
                note(job, f"Could not reach Kaggle (attempt {attempt}), trying again in a minute")
            time.sleep(60)
        version = re.search(r"Kernel version (\d+) successfully pushed", out)
        for job in ready:
            note(job, (out.splitlines()[-1] if out else f"push exit {code}") + f" (GPU: {GPUS[machine]}"
                 + (f", batch of {len(ready)} reels)" if len(ready) > 1 else ")"))
            if version:  # remember which Kaggle run is ours, so we download exactly its results later
                update(job, kaggle_version=int(version[1]))
        return not (code or "error" in out.lower()), out

    machine = "NvidiaTeslaT4"
    ok, out = push(machine)
    if not ok:
        for job in ready:
            fail(job, RuntimeError("Kaggle did not accept the job: " + out[-300:]))
        return
    started = pushed = time.time()
    for job in ready:
        update(job, stage="Sent to Kaggle", kaggle_started=started, kaggle_url=f"https://www.kaggle.com/code/{ref}",
               gpu=GPUS[machine])

    time.sleep(45)
    watch(ready, folder, ref, machine, started, pushed, push)


def watch(ready, folder, ref, machine, started, pushed, push=None):
    """Follow one Kaggle run until it ends, then hand out its results. Also used after a restart of the website,
    to pick up a run that was already on Kaggle (push=None then: no GPU switch from here)."""
    stale_since = None  # when Kaggle first looked finished without results from this run
    while True:
        status = kaggle_status(ref)
        settled = time.time() - pushed > 900  # right after a push, Kaggle can still report the previous run
        if status == "queued":
            waited = (time.time() - pushed) / 60
            stage = f"Waiting for a free Kaggle GPU ({GPUS[machine]}, {waited:.0f} min)"
            if push and GPU_WAIT_MINUTES and machine == "NvidiaTeslaT4" and waited >= GPU_WAIT_MINUTES:
                for job in ready:
                    note(job, f"No T4 GPU after {waited:.0f} min, trying the P100 instead (uses the smaller gemma3:12b model)")
                ok, out = push("NvidiaTeslaP100")
                if ok:
                    machine, pushed = "NvidiaTeslaP100", time.time()
                    stage = "Waiting for a free Kaggle GPU (P100, 0 min)"
                else:
                    for job in ready:
                        note(job, "P100 push failed, staying in the T4 queue: " + out[-200:])
                    machine = "T4 only"  # don't try again
        elif status == "running":
            stage = f"Making the reel on Kaggle ({GPUS.get(machine, 'T4 x2')})"
        else:
            stage = None
        for job in ready:
            changes = {"kaggle_status": status, "kaggle_minutes": round((time.time() - started) / 60, 1),
                       "gpu": GPUS.get(machine, "T4 x2")}
            if stage:
                changes["stage"] = stage
            update(job, **changes)
        if all(j.get("stopped") for j in ready):
            return  # everything in this run was stopped from the website; stop watching Kaggle
        if status in ("complete", "error", "cancel"):
            out_dir = folder.parent / "result"
            download_results(ready, ref, out_dir)
            if not (out_dir / ready[0]["id"]).exists():
                # No results from OUR run yet. Kaggle's status can belong to an older (e.g. cancelled) run while ours
                # is still queued (tonight: 30 min), so only give up once it has looked finished-without-results for an hour.
                if stale_since is None:
                    stale_since = time.time()
                    for job in ready:
                        note(job, f"Kaggle says '{status}' but has no results from this run yet, waiting...")
                if not settled or time.time() - stale_since < 60 * 60:
                    time.sleep(30)
                    continue
            for job in ready:
                if not job.get("stopped"):
                    collect(job, out_dir / job["id"], status, out_dir / "batch_log.txt")
            return
        stale_since = None  # Kaggle shows our run as queued/running again
        if time.time() - started > 6 * 3600:
            for job in ready:
                fail(job, RuntimeError("Kaggle took more than 6 hours; giving up."))
            return
        time.sleep(30)


def download_results(ready, ref, out_dir, tries=3):
    """Download this run's results: exactly our Kaggle version (not just "the newest run"), all pages (a batch of
    5 reels is ~27 files; the tool shows 20 per page by default), and check that every reel came back. If the
    download was cut short (an internet drop), download again before anyone gets marked as failed."""
    version = ready[0].get("kaggle_version")
    target = f"{ref}/{version}" if version else ref
    for attempt in range(1, tries + 1):
        shutil.rmtree(out_dir, ignore_errors=True)
        kaggle("kernels", "output", target, "-p", str(out_dir), "-o", "-q", "--page-size", "200", timeout=900)
        if not (out_dir / ready[0]["id"]).exists():
            return  # nothing from our run yet (Kaggle may still show an older run): the caller waits and asks again
        missing = [j["topic"] for j in ready if not (out_dir / j["id"] / "result.json").exists()]
        if not missing:
            return
        if attempt < tries:
            for job in ready:
                note(job, f"The download from Kaggle was incomplete ({len(missing)} reel(s) missing), downloading again")
            time.sleep(30)


def collect_hooks(job, res_dir, status):
    """New hooks came back: add them to the reel they belong to (after the ones already shown)."""
    res_file, hooks_file = res_dir / "result.json", res_dir / "hooks.json"
    res = json.loads(res_file.read_text(encoding="utf-8")) if res_file.exists() else {}
    if res.get("status") != "done" or not hooks_file.exists():
        fail(job, RuntimeError("Writing new hooks failed: " + res.get("error", f"no result ({status})")))
        try:
            update(load(job["parent_id"]), hooks_job=None)
        except FileNotFoundError:
            pass
        return
    new = json.loads(hooks_file.read_text(encoding="utf-8")).get("hooks", [])
    with lock:
        parent = json.loads(job_path(job["parent_id"]).read_text(encoding="utf-8"))
    script = parent.get("script") or {}
    options = script.setdefault("hook_options", [])
    known = {h["spoken"].lower() for h in options}
    first_new = len(options)
    options += [h for h in new if h["spoken"].lower() not in known]
    pscript = JOBS / parent["id"] / "script.json"
    if pscript.exists():  # keep the reel's script file in step, so "Remake with this hook" finds them
        on_disk = json.loads(pscript.read_text(encoding="utf-8"))
        on_disk["hook_options"] = options
        pscript.write_text(json.dumps(on_disk, indent=2, ensure_ascii=False), encoding="utf-8")
    update(parent, script=script, hooks_job=None, new_hooks_from=first_new)
    note(parent, f"{len(options) - first_new} new hooks written")
    update(job, status="done", stage="Done", timings=res.get("timings", {}), model=res.get("model"))


def collect(job, res_dir, status, batch_log):
    if job.get("kind") == "hooks":
        return collect_hooks(job, res_dir, status)
    jdir = JOBS / job["id"]
    for name in ("reel.mp4", "script.json", "log.txt", "photo_credits.txt"):
        if (res_dir / name).exists():
            shutil.copy(res_dir / name, jdir / name)
    if not (jdir / "log.txt").exists() and batch_log.exists():
        shutil.copy(batch_log, jdir / "log.txt")
    res_file = res_dir / "result.json"
    res = json.loads(res_file.read_text(encoding="utf-8")) if res_file.exists() else {}
    if res.get("status") == "done" and (jdir / "reel.mp4").exists():
        script = json.loads((jdir / "script.json").read_text(encoding="utf-8"))
        update(job, status="done", stage="Done", script=script, timings=res.get("timings", {}),
               setup_timings=res.get("setup_timings", {}), model=res.get("model"),
               total_minutes=round((time.time() - job["created"]) / 60, 1))
    else:
        fail(job, RuntimeError("The Kaggle run failed: " + res.get("error", f"no result ({status})")))


def next_job(block=True):
    """The next reel in line, skipping any deleted while waiting (stopped, then deleted). None if the line is empty."""
    while True:
        try:
            jid = work_queue.get(block=block)
        except queue.Empty:
            return None
        if job_path(jid).exists():
            return load(jid)


RESUME = []  # after a restart: [jobs of the Kaggle run that was still going], picked up before anything new is sent


def worker_loop():
    if RESUME:  # first finish following the run that was on Kaggle when the website closed
        ready = RESUME[0]
        try:
            ref = f"{kaggle_user()}/{SLUG}"
            for job in ready:
                note(job, "The website restarted; following this reel's Kaggle run again")
            watch(ready, JOBS / ready[0]["id"] / "kaggle", ref, "NvidiaTeslaT4", ready[0]["kaggle_started"],
                  ready[0]["kaggle_started"])
        except Exception as e:
            traceback.print_exc()
            for job in ready:
                if load(job["id"]).get("status") not in ("done", "error", "stopped"):
                    fail(job, e)
    while True:  # nothing below may end this loop: if it did, no reel would ever start again
        jobs = []
        try:
            jobs = [next_job()]
            time.sleep(3)  # a moment for more topics to arrive, so they can share the run
            while len(jobs) < MAX_BATCH:
                job = next_job(block=False)
                if job is None:
                    break
                jobs.append(job)
            run_batch(jobs)
        except Exception as e:
            traceback.print_exc()
            for job in jobs:
                try:
                    if job_path(job["id"]).exists() and load(job["id"]).get("status") not in ("done", "error", "stopped"):
                        fail(job, e)
                except Exception:
                    traceback.print_exc()


# ---------- web pages ----------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    return jsonify({"kaggle_user": kaggle_user(), "queue": work_queue.qsize()})


@app.post("/api/jobs")
def create_job():
    """One job per topic line; several lines share the other settings and are made in one Kaggle batch."""
    topics = [t.strip() for t in request.form.get("topic", "").splitlines() if t.strip()][:MAX_BATCH * 2]
    if not topics:
        return jsonify({"error": "Please type a topic."}), 400
    pdfs = [f.read() for f in request.files.getlist("pdfs") if f and f.filename.lower().endswith(".pdf")]
    ids = []
    for topic in topics:
        ids.append(new_job(topic, pdfs))
    return jsonify({"id": ids[0], "ids": ids})


def new_job(topic, pdfs, form=None, extra=None):
    """form: the Create page's fields (a plain dict when Discover makes the reel). extra: fields added to the job."""
    form = request.form if form is None else form
    jid = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    jdir = JOBS / jid
    (jdir / "uploads").mkdir(parents=True)
    for k, data in enumerate(pdfs):
        (jdir / "uploads" / f"{k:02d}.pdf").write_bytes(data)
    links = [l.strip() for l in form.get("links", "").splitlines() if l.strip().startswith("http")]
    job = {"id": jid, "topic": topic, "news": True,  # always News mode: every reel is researched from recent articles
           "only_sources": form.get("only_sources") == "on", "links": links,
           "press_photos": form.get("press_photos") == "on",
           "hook_idea": form.get("hook_idea", "").strip()[:300],
           "format": form.get("format", "9:16") if form.get("format") in ("9:16", "1:1", "16:9") else "9:16",
           "voice": re.sub(r"[^\w-]", "", form.get("voice", ""))[:60],
           "music": form.get("music", "off") if form.get("music") in MUSIC_MOODS else "off",
           "instructions": form.get("instructions", "").strip()[:1500],
           "pasted": form.get("pasted", ""), "pdf_count": len(list((jdir / "uploads").glob("*.pdf"))),
           "created": time.time(), "status": "queued", "stage": "Waiting in line", "log": [], **(extra or {})}
    save(job)
    work_queue.put(jid)
    return jid


@app.get("/api/jobs")
def list_jobs():
    jobs = []
    for p in sorted(JOBS.glob("*/job.json"), reverse=True)[:300]:
        j = json.loads(p.read_text(encoding="utf-8"))
        j["has_video"] = (p.parent / "reel.mp4").exists()
        j.pop("pasted", None)
        jobs.append(j)
    return jsonify(jobs)


@app.get("/jobs/<jid>/<name>")
def job_file(jid, name):
    if name not in ("reel.mp4", "log.txt", "photo_credits.txt") or not re.fullmatch(r"[\w-]+", jid):
        abort(404)
    return send_from_directory(JOBS / jid, name, as_attachment=request.args.get("download") == "1",
                               download_name=f"reel-{jid}.mp4" if name == "reel.mp4" else name)


@app.post("/api/jobs/<jid>/new-hooks")
def new_hooks(jid):
    """Write a fresh set of hooks for a finished reel (a short Kaggle run: no new research, no new video)."""
    if not re.fullmatch(r"[\w-]+", jid) or not job_path(jid).exists():
        abort(404)
    parent = load(jid)
    if parent.get("status") != "done" or not parent.get("script"):
        return jsonify({"error": "New hooks can only be written for a finished reel."}), 409
    if parent.get("hooks_job") and job_path(parent["hooks_job"]).exists() \
            and load(parent["hooks_job"]).get("status") in ("queued", "working"):
        return jsonify({"error": "New hooks are already being written for this reel."}), 409
    new_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    (JOBS / new_id).mkdir(parents=True)
    job = {k: parent[k] for k in ("topic", "news", "only_sources", "instructions", "hook_idea") if k in parent}
    job.update(id=new_id, kind="hooks", parent_id=jid, links=[], pasted="", created=time.time(), status="queued",
               stage="Waiting in line", log=[])
    save(job)
    update(parent, hooks_job=new_id)
    work_queue.put(new_id)
    return jsonify({"id": new_id})


@app.post("/api/jobs/<jid>/remake")
def remake(jid):
    """Make the reel again with the same settings and sources. Optional "hook": index into the parent's top hooks."""
    if not re.fullmatch(r"[\w-]+", jid) or not job_path(jid).exists():
        abort(404)
    parent = load(jid)
    options = (parent.get("script") or {}).get("hook_options") or parent.get("hook_options") or []
    body = request.get_json(silent=True) or {}
    forced = None
    if body.get("hook") is not None:
        try:
            forced = options[int(body["hook"])]
        except (ValueError, IndexError, TypeError):
            return jsonify({"error": "That hook is not available for this reel."}), 400
    new_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    (JOBS / new_id / "uploads").mkdir(parents=True)
    for pdf in (JOBS / jid / "uploads").glob("*.pdf"):
        shutil.copy(pdf, JOBS / new_id / "uploads" / pdf.name)
    keep = ("topic", "news", "only_sources", "links", "press_photos", "hook_idea", "format", "voice", "music",
            "instructions", "pasted", "pdf_count")
    job = {k: parent[k] for k in keep if k in parent}
    job.update(id=new_id, parent_id=jid, news=True, forced_hook=forced, hook_options=options if forced else None,
               created=time.time(), status="queued", stage="Waiting in line", log=[])
    save(job)
    work_queue.put(new_id)
    return jsonify({"id": new_id})


def remove_job(jid):
    """Delete a finished or failed reel and all its files. Returns an error message, or "" when deleted."""
    if not re.fullmatch(r"[\w-]+", str(jid)) or not job_path(jid).exists():
        return "not found"
    if load(jid).get("status") in ("queued", "working"):
        return "still being made"
    shutil.rmtree(JOBS / jid, ignore_errors=True)
    return ""


@app.post("/api/jobs/<jid>/stop")
def stop_job(jid):
    """Stop a reel that is waiting or being made. The site stops working on it at once; a run already sent to
    Kaggle can't be cancelled from here (Kaggle's tools don't expose it), so the page links to Kaggle's Stop button."""
    if not re.fullmatch(r"[\w-]+", jid) or not job_path(jid).exists():
        abort(404)
    with lock:
        job = load(jid)
        if job.get("status") not in ("queued", "working"):
            return jsonify({"error": "This reel is not being made."}), 409
        on_kaggle = bool(job.get("kaggle_started"))
        others = [b for b in job.get("batch", []) if b != jid and job_path(b).exists()
                  and load(b).get("status") in ("queued", "working")]
        job.update(stopped=True, status="stopped", stage="Stopped by you",
                   error="You stopped this reel." + (" Kaggle may still run it; stop it on the Kaggle page to save GPU hours."
                                                     if on_kaggle and not others else ""))
        job.setdefault("log", []).append(f"{datetime.now():%H:%M:%S}  stopped from the website")
        job_path(jid).write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
    return jsonify({"ok": True, "on_kaggle": on_kaggle and not others, "kaggle_url": job.get("kaggle_url", ""),
                    "shared_with": len(others)})


@app.delete("/api/jobs/<jid>")
def delete_job(jid):
    err = remove_job(jid)
    if err == "not found":
        abort(404)
    if err:
        return jsonify({"error": "This reel is still being made."}), 409
    return jsonify({"ok": True})


@app.post("/api/jobs/delete")
def delete_jobs():
    """Delete several reels at once. Reels still being made are skipped."""
    ids = (request.get_json(silent=True) or {}).get("ids") or []
    deleted, skipped = [], []
    for jid in ids[:500]:
        (skipped if remove_job(jid) else deleted).append(jid)
    return jsonify({"deleted": deleted, "skipped": skipped})


@app.get("/api/download")
def download_many():
    """One zip with the chosen reels' videos, named after their topics."""
    import io, zipfile
    ids = [i for i in request.args.get("ids", "").split(",") if re.fullmatch(r"[\w-]+", i)]
    buf, names = io.BytesIO(), set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # mp4 is already compressed
        for jid in ids:
            video = JOBS / jid / "reel.mp4"
            if not video.exists():
                continue
            topic = load(jid).get("topic", jid) if job_path(jid).exists() else jid
            name = re.sub(r"[^\w -]", "", topic)[:60].strip() or jid
            while name + ".mp4" in names:
                name += " " + jid[-4:]
            names.add(name + ".mp4")
            z.write(video, name + ".mp4")
    if not names:
        return jsonify({"error": "None of those reels has a video."}), 404
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"reels-{datetime.now():%Y%m%d-%H%M}.zip")


PREVIEWS = HERE / "voice_previews"
PREVIEW_TEXT = "Hello! This is how your reel will sound."


@app.get("/api/voice-preview/<voice>")
def voice_preview(voice):
    """A short sample of an edge-tts voice, made once per voice and then reused (same speed as the reels)."""
    if not re.fullmatch(r"[a-z]{2}-[A-Z]{2}-[A-Za-z]+Neural", voice):
        abort(404)
    path = PREVIEWS / f"{voice}.mp3"
    if not path.exists():
        import asyncio, edge_tts
        PREVIEWS.mkdir(exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            asyncio.run(edge_tts.Communicate(PREVIEW_TEXT, voice, rate="+10%").save(str(tmp)))
            tmp.replace(path)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            return jsonify({"error": f"Could not make the voice sample ({e}). Is the internet on?"}), 502
    return send_from_directory(PREVIEWS, path.name, mimetype="audio/mpeg", max_age=86400)


KEY_NAMES = ("flickr_api_key", "pexels_api_key", "pixabay_api_key")


@app.get("/api/settings")
def get_settings():
    """Which photo/video keys are set (never the keys themselves)."""
    cfg = settings()
    return jsonify({k: bool(cfg.get(k)) for k in KEY_NAMES})


@app.post("/api/settings")
def save_settings():
    cfg, body = settings(), request.get_json(silent=True) or {}
    for k in KEY_NAMES:
        v = str(body.get(k, "")).strip()
        if v == "-":
            cfg[k] = ""          # remove the key
        elif v:
            cfg[k] = re.sub(r"[^\w-]", "", v)[:100]
    CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return get_settings()


# ---------- Instagram: post a finished reel after you approve it (on this laptop, no GPU) ----------

post_now = threading.Event()        # wakes the posting thread when you approve a post
POSTING = {"busy": None}            # id of the reel being posted right now
RETRY_MINUTES = (5, 15, 60)         # waits between tries when Instagram or the internet has trouble


def ig_settings():
    return settings().get("instagram") or {}


def save_config(cfg):
    tmp = CONFIG.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tmp.replace(CONFIG)


def ig_update(jid, **changes):
    """Change a reel's posting state, reading the newest copy from disk so nothing else on the reel is lost."""
    with lock:
        job = load(jid)
        ig = job.setdefault("ig", {})
        ig.update(changes)
        for k in [k for k, v in ig.items() if v is None]:
            del ig[k]
        job_path(jid).write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
        return job


def due_posts(now=None):
    """Reels approved for posting whose time has come, oldest first. A post that was interrupted (site closed
    mid-post) is picked up again; it resumes from its Instagram container, so it is never posted twice."""
    now = now or time.time()
    due = []
    for p in JOBS.glob("*/job.json"):
        try:
            ig = json.loads(p.read_text(encoding="utf-8")).get("ig") or {}
        except (OSError, ValueError):
            continue
        if ig.get("state") == "posting" or (ig.get("state") == "scheduled" and ig.get("at", 0) <= now):
            due.append((ig.get("at", 0), p.parent.name))
    return [jid for _, jid in sorted(due)]


def post_one(jid):
    cfg = ig_settings()
    job = load(jid)
    ig = job.get("ig") or {}
    if not cfg.get("token"):
        return ig_update(jid, state="failed", stage=None, error="Instagram is not connected. Connect it in Settings.")
    video = JOBS / jid / "reel.mp4"
    if not video.exists():
        return ig_update(jid, state="failed", stage=None, error="This reel has no video file.")
    tries = ig.get("tries", 0) + 1
    ig_update(jid, state="posting", stage="Starting", tries=tries, error=None)
    note(load(jid), f"posting to Instagram (try {tries})")
    try:
        res = instagram.publish(cfg, video, ig.get("caption", ""), container=ig.get("container"),
                                on_step=lambda s: ig_update(jid, stage=s),
                                on_container=lambda c: ig_update(jid, container=c), ai=ig.get("ai", True))
    except instagram.IGError as e:
        if e.retry and tries <= len(RETRY_MINUTES):
            wait = RETRY_MINUTES[tries - 1]
            ig_update(jid, state="scheduled", at=time.time() + wait * 60, stage=None,
                      error=f"{e} Trying again in {wait} min.")
        else:
            ig_update(jid, state="failed", stage=None, error=str(e))
        note(load(jid), f"Instagram post failed: {e}")
        return
    except Exception as e:  # a bug, not Instagram: stop instead of retrying forever
        ig_update(jid, state="failed", stage=None, error=f"{type(e).__name__}: {e}")
        note(load(jid), "Instagram post failed:\n" + traceback.format_exc()[-1500:])
        return
    ig_update(jid, state="posted", stage=None, error=None, posted_at=time.time(),
              media_id=res.get("media_id"), permalink=res.get("permalink"))
    note(load(jid), "posted to Instagram " + (res.get("permalink") or ""))


def post_loop():
    while True:
        post_now.wait(20)
        post_now.clear()
        for jid in due_posts():
            POSTING["busy"] = jid
            try:
                post_one(jid)
            except Exception:
                traceback.print_exc()
            finally:
                POSTING["busy"] = None


@app.get("/api/instagram")
def get_instagram():
    cfg = ig_settings()
    return jsonify({"connected": bool(cfg.get("token")), "username": cfg.get("username", ""),
                    "page": cfg.get("page", ""), "connected_at": cfg.get("connected_at"),
                    "hashtags": settings().get("default_hashtags", "")})


@app.post("/api/instagram/connect")
def ig_connect():
    body = request.get_json(silent=True) or {}
    token, app_id, secret = (str(body.get(k, "")).strip() for k in ("token", "app_id", "app_secret"))
    if not (token and app_id and secret):
        return jsonify({"error": "Paste the token, the App ID and the App Secret."}), 400
    if not re.fullmatch(r"\d{5,25}", app_id):
        return jsonify({"error": "The App ID is a long number. Copy it from App settings > Basic."}), 400
    try:
        got = instagram.connect(token, app_id, secret)
    except instagram.IGError as e:
        return jsonify({"error": str(e)}), 400
    cfg = settings()
    cfg["instagram"] = {**got, "connected_at": time.time()}   # the App Secret is not kept: the Page token never expires
    save_config(cfg)
    return get_instagram()


@app.post("/api/instagram/disconnect")
def ig_disconnect():
    cfg = settings()
    cfg.pop("instagram", None)
    save_config(cfg)
    return get_instagram()


@app.post("/api/instagram/hashtags")
def ig_hashtags():
    tags = str((request.get_json(silent=True) or {}).get("hashtags", ""))
    tags = " ".join("#" + t.lstrip("#") for t in re.findall(r"#?[\w]+", tags))[:300]
    cfg = settings()
    cfg["default_hashtags"] = tags
    save_config(cfg)
    return get_instagram()


@app.post("/api/instagram/check")
def ig_check():
    cfg = ig_settings()
    if not cfg.get("token"):
        return jsonify({"error": "Instagram is not connected."}), 409
    try:
        name = instagram.account(cfg)
    except instagram.IGError as e:
        return jsonify({"error": str(e)}), 502
    left = instagram.posts_left(cfg)
    return jsonify({"username": name, "used": left[0] if left else None, "allowed": left[1] if left else None})


@app.post("/api/jobs/<jid>/post")
def post_reel(jid):
    """Approve a reel for Instagram: post now, or at a time you pick ("at": unix seconds)."""
    if not re.fullmatch(r"[\w-]+", jid) or not job_path(jid).exists():
        abort(404)
    body = request.get_json(silent=True) or {}
    job = load(jid)
    ig = job.get("ig") or {}
    if job.get("status") != "done" or not (JOBS / jid / "reel.mp4").exists():
        return jsonify({"error": "Only a finished reel can be posted."}), 409
    if not ig_settings().get("token"):
        return jsonify({"error": "Connect Instagram in Settings first."}), 409
    if ig.get("state") in ("posting", "posted"):
        return jsonify({"error": "This reel is already " + ("being posted." if ig["state"] == "posting" else "posted.")}), 409
    caption = str(body.get("caption", "")).strip()[:2200]
    ai = body.get("ai", True) is not False  # Instagram's "AI info" label, on unless turned off
    container = ig.get("container")
    if container and (caption != ig.get("caption") or ai != ig.get("ai", True)):  # made with other text or label: don't reuse it
        try:
            if instagram.container_status(ig_settings(), container).get("status_code") == "PUBLISHED":
                ig_update(jid, state="posted", posted_at=time.time(), error=None, stage=None)
                return jsonify({"error": "This reel was already posted by an earlier try."}), 409
        except instagram.IGError:
            pass
        container = None
    try:
        at = float(body.get("at") or 0)
    except (TypeError, ValueError):
        return jsonify({"error": "That time is not valid."}), 400
    now = time.time()
    if at and at < now - 60:
        return jsonify({"error": "That time has already passed."}), 400
    if at > now + 75 * 86400:
        return jsonify({"error": "Pick a time within the next 75 days."}), 400
    ig_update(jid, state="scheduled", at=max(at, now), caption=caption, ai=ai, approved_at=now, tries=0, error=None,
              stage=None, container=container)
    note(load(jid), "approved for Instagram" + (f" at {datetime.fromtimestamp(at):%d %b %H:%M}" if at > now + 60 else ""))
    if at <= now + 60:
        post_now.set()
    return jsonify({"ok": True})


@app.post("/api/jobs/<jid>/post/cancel")
def cancel_post(jid):
    if not re.fullmatch(r"[\w-]+", jid) or not job_path(jid).exists():
        abort(404)
    ig = load(jid).get("ig") or {}
    if ig.get("state") not in ("scheduled", "failed") or POSTING["busy"] == jid:
        return jsonify({"error": "This post can't be cancelled now."}), 409
    with lock:  # keep the container id: if it was already made, a later post picks it up instead of uploading twice
        job = load(jid)
        job["ig"] = {k: v for k, v in ig.items() if k in ("container", "caption", "ai")}
        job_path(jid).write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
    note(load(jid), "Instagram post cancelled")
    return jsonify({"ok": True})


# ---------- Discover: find stories worth a reel (on this laptop, no GPU) ----------

DISCOVER = HERE / "discover.json"   # the current story list, when the last scan ran, and stories you made or skipped
DISCOVER_DEFAULTS = {"every_min": 60, "show": 10, "auto": False, "auto_count": 2, "auto_hours": 4, "min_score": 60}
SCAN_CHOICES = (15, 30, 60, 120, 180, 360, 720)        # minutes
SHOW_CHOICES = (5, 10, 15, 20, 30)
AUTO_HOURS = (1, 2, 3, 4, 6, 8, 12, 24)
DECIDED_DAYS = 3                    # a story you made or skipped is not shown again for this long (it's stale by then)
disc_lock = threading.Lock()
scan_now = threading.Event()
DISC = {"scanning": False, "progress": ""}  # live scan state (not saved)


def disc_settings():
    s = {**DISCOVER_DEFAULTS, **(settings().get("discover") or {})}
    return s


def disc_state():
    try:
        st = json.loads(DISCOVER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    st.setdefault("stories", []); st.setdefault("decided", {})
    return st


def disc_save(st):
    tmp = DISCOVER.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    tmp.replace(DISCOVER)


def made_topics(days=7):
    """Topics of reels from the last week, so Discover doesn't suggest a story you already have a reel about."""
    cutoff, out = time.time() - days * 86400, []
    for p in JOBS.glob("*/job.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if j.get("created", 0) >= cutoff and j.get("kind") != "hooks" and j.get("status") not in ("error", "stopped"):
            out.append(j["topic"])
    return out


def run_scan():
    DISC.update(scanning=True, progress="Reading the news")
    started = time.time()
    with disc_lock:
        st = disc_state()
    old, decided = {s["id"]: s for s in st["stories"]}, dict(st["decided"])
    try:
        stories, problems = discover.scan(old, decided, made_topics(), log=lambda m: DISC.update(progress=m))
        err = ""
    except Exception as e:
        traceback.print_exc()
        stories, problems, err = None, [], str(e)[:300]
    with disc_lock:
        st = disc_state()  # you may have made or skipped stories while the scan ran
        if stories is not None:
            st["stories"] = [s for s in stories if s["id"] not in st["decided"]]
        st["decided"] = {k: v for k, v in st["decided"].items() if time.time() - v.get("at", 0) < DECIDED_DAYS * 86400}
        st.update(last_scan=started, last_error=err, problems=problems[:5])
        disc_save(st)
    DISC.update(scanning=False, progress="")


def reel_options():
    """Reels made from Discover use the same voice, format, music and photo choice as your most recent own reel."""
    form = {"format": "9:16", "voice": "", "music": "off", "press_photos": "on"}
    for p in sorted(JOBS.glob("*/job.json"), reverse=True)[:40]:
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if j.get("kind") != "hooks" and not j.get("from_discover"):
            form.update(format=j.get("format", "9:16"), voice=j.get("voice", ""), music=j.get("music", "off"),
                        press_photos="on" if j.get("press_photos") else "")
            break
    return form


def decide(ids, status, how="you"):
    """status "made": each story becomes a reel (its news links go along as sources); "skipped": it's hidden.
    Either way the story never comes back. Returns the new reel ids."""
    made, form = [], reel_options()
    with disc_lock:
        st = disc_state()
        for sid in ids:
            s = next((x for x in st["stories"] if x["id"] == sid), None)
            if not s:
                continue
            rec = {"status": status, "core": s["core"], "at": time.time(), "title": s["title"], "score": s["score"], "how": how}
            if status == "made":
                rec["job"] = new_job(s["title"], [], {**form, "links": "\n".join(s.get("reel_links", []))},
                                     {"from_discover": sid, "discover_score": s["score"], "auto": how == "auto"})
                made.append(rec["job"])
            st["decided"][sid] = rec
            st["stories"].remove(s)
        disc_save(st)
    return made


def auto_due(s, st):
    if not s["auto"]:
        return None
    return (st.get("auto_window_start") or time.time()) + s["auto_hours"] * 3600


def run_auto():
    """End of an Auto window: make the best stories found during it (at most x, only those above your minimum score)."""
    s = disc_settings()
    with disc_lock:
        st = disc_state()
        picks = [x["id"] for x in st["stories"] if x["score"] >= s["min_score"]][: s["auto_count"]]
        st["auto_window_start"] = time.time()
        st["auto_last"] = {"at": time.time(), "made": len(picks)}
        disc_save(st)
    if picks:
        decide(picks, "made", how="auto")
    print(f"Discover auto mode: made {len(picks)} reel(s)")


def discover_loop():
    """Scan every few minutes/hours (your choice) or when you press Scan now; in Auto mode, make reels at each window's end."""
    while True:
        try:
            s, now = disc_settings(), time.time()
            with disc_lock:
                st = disc_state()
                due = auto_due(s, st)
                if s["auto"] and (not st.get("auto_window_start") or now - due > 1800):
                    # Auto was just switched on, or the website was closed when a window ended: start a fresh window
                    st["auto_window_start"] = now
                    disc_save(st)
                    due = auto_due(s, st)
            wake = min((st.get("last_scan") or 0) + s["every_min"] * 60, due or float("inf"))
            if wake > now and not scan_now.wait(timeout=min(wake - now, 30)):
                continue  # not time yet (settings are looked at again every 30 s)
            scan_now.clear()
            run_scan()
            s = disc_settings()
            if s["auto"] and time.time() >= (auto_due(s, disc_state()) or float("inf")):
                run_auto()
        except Exception:
            traceback.print_exc()
            DISC.update(scanning=False, progress="")
            time.sleep(60)


def gpu_minutes_per_reel():
    """Average Kaggle minutes per reel over your recent runs (several reels share one run)."""
    runs = {}
    for p in JOBS.glob("*/job.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if j.get("status") == "done" and j.get("kaggle_minutes") and j.get("kaggle_started"):
            runs[j["kaggle_started"]] = (j["kaggle_minutes"], len(j.get("batch") or [j["id"]]))
    recent = [runs[k] for k in sorted(runs)[-15:]]
    return sum(m for m, _ in recent) / sum(n for _, n in recent) if recent else 8.0


@app.get("/api/discover")
def get_discover():
    s, st, now = disc_settings(), disc_state(), time.time()
    keep = ("id", "title", "summary", "site_count", "sites", "newest", "first_seen", "links", "reel_links", "trend",
            "ai", "parts", "score", "reasons")
    stories = [{k: x.get(k) for k in keep} for x in st["stories"][: s["show"]]]
    per_reel = gpu_minutes_per_reel()
    due = auto_due(s, st)
    return jsonify({"settings": s, "stories": stories, "total": len(st["stories"]),
                    "above_min": sum(x["score"] >= s["min_score"] for x in st["stories"]),
                    "last_scan": st.get("last_scan"), "next_scan": (st.get("last_scan") or now) + s["every_min"] * 60,
                    "scanning": DISC["scanning"], "progress": DISC["progress"], "error": st.get("last_error", ""),
                    "problems": st.get("problems", []), "auto_next": due, "auto_last": st.get("auto_last"),
                    "gpu_per_reel": round(per_reel, 1),
                    "gpu_week_hours": round(s["auto_count"] * 168 / s["auto_hours"] * per_reel / 60, 1),
                    "choices": {"every_min": SCAN_CHOICES, "show": SHOW_CHOICES, "auto_hours": AUTO_HOURS,
                                "auto_count": list(range(1, MAX_BATCH + 1))},
                    "made": sorted(({"title": v["title"], "job": v.get("job"), "how": v.get("how"), "at": v["at"],
                                     "score": v.get("score")} for v in st["decided"].values() if v["status"] == "made"),
                                   key=lambda v: -v["at"])[:10]})


@app.post("/api/discover/scan")
def discover_scan():
    scan_now.set()
    return jsonify({"ok": True})


@app.post("/api/discover/settings")
def discover_settings():
    body, cfg = request.get_json(silent=True) or {}, settings()
    s = {**DISCOVER_DEFAULTS, **(cfg.get("discover") or {})}

    def pick(key, allowed):
        try:
            v = int(body[key])
        except (KeyError, TypeError, ValueError):
            return
        if v in allowed:
            s[key] = v
    pick("every_min", SCAN_CHOICES); pick("show", SHOW_CHOICES); pick("auto_hours", AUTO_HOURS)
    pick("auto_count", range(1, MAX_BATCH + 1)); pick("min_score", range(0, 101))
    if "auto" in body:
        turned_on = bool(body["auto"]) and not s["auto"]
        s["auto"] = bool(body["auto"])
        with disc_lock:
            st = disc_state()
            st["auto_window_start"] = time.time() if s["auto"] else None
            disc_save(st)
        if turned_on:
            print("Discover auto mode switched on")
    cfg["discover"] = s
    CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return get_discover()


@app.post("/api/discover/make")
def discover_make():
    ids = [str(i) for i in (request.get_json(silent=True) or {}).get("ids", [])][:MAX_BATCH * 2]
    made = decide(ids, "made")
    return jsonify({"made": made})


@app.post("/api/discover/skip")
def discover_skip():
    ids = [str(i) for i in (request.get_json(silent=True) or {}).get("ids", [])][:100]
    decide(ids, "skipped")
    return jsonify({"ok": True})


def resume_unfinished():
    """After a restart: reels still waiting in line go back in line (nothing was started for them yet);
    the most recent Kaggle run that was still going is followed again (RESUME); anything else that was being made
    is marked stopped, so you can remake it."""
    on_kaggle = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(JOBS.glob("*/job.json"))]
    on_kaggle = [j for j in on_kaggle if j.get("status") == "working" and j.get("kaggle_started") and not j.get("stopped")]
    if on_kaggle:
        latest = max(j["kaggle_started"] for j in on_kaggle)
        batch = [j for j in on_kaggle if j["kaggle_started"] == latest]
        batch.sort(key=lambda j: (j.get("batch") or [j["id"]]).index(j["id"]) if j["id"] in (j.get("batch") or []) else 0)
        RESUME.append(batch)
    resumed = {j["id"] for b in RESUME for j in b}
    for p in sorted(JOBS.glob("*/job.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        if j["id"] in resumed:
            continue
        if j.get("status") == "queued" and not j.get("stopped"):
            work_queue.put(j["id"])
        elif j.get("status") == "working" and not j.get("kaggle_started") and not j.get("stopped"):
            # it was still being researched on this laptop: nothing was sent to Kaggle yet, so just put it back in line
            j.update(status="queued", stage="Waiting in line")
            j.setdefault("log", []).append(f"{datetime.now():%H:%M:%S}  The website restarted during research; back in line")
            p.write_text(json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8")
            work_queue.put(j["id"])
        elif j.get("status") in ("queued", "working"):
            j.update(status="stopped", stopped=True, stage="Stopped",
                     error="The website was closed before this reel finished." +
                           (" Kaggle may still run it; stop it on the Kaggle page to save GPU hours." if j.get("kaggle_started") else ""))
            p.write_text(json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8")


def already_running(port=5050):
    """Windows lets a second copy listen on the same port silently; two copies would fight over the same reels."""
    import socket
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


if __name__ == "__main__":
    if already_running():
        print("Reel Studio is already running: open http://127.0.0.1:5050\n"
              "To restart it, close the other Reel Studio window first.")
        sys.exit(1)
    JOBS.mkdir(exist_ok=True)
    resume_unfinished()
    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=discover_loop, daemon=True).start()
    threading.Thread(target=post_loop, daemon=True).start()
    print("Reel Studio is running: open http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
