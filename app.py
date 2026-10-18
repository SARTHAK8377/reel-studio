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

def kaggle(*args, timeout=300):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return r.returncode, (r.stdout + r.stderr).strip()


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


GPU_WAIT_MINUTES = 15  # queued longer than this on T4 x2: try the single P100 instead
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


def research_job(job):
    """Research on this laptop. Returns the payload the Kaggle worker needs for this reel."""
    jdir = JOBS / job["id"]
    update(job, stage="Researching", status="working")
    pdfs = sorted((jdir / "uploads").glob("*.pdf"))
    sources, problems = research.gather(job["topic"], job["links"], pdfs, job["pasted"], web=job["news"],
                                        log=lambda m: note(job, m), wiki=not job["only_sources"])
    for p in problems:
        note(job, "skipped: " + p)
    if job["only_sources"] and not sources:
        raise RuntimeError("You chose 'only my sources', but none of them could be read.")
    (jdir / "sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
    note(job, f"{len(sources)} source(s) ready")
    pool, problems = research.photo_pool(job["topic"], sources, job.get("press_photos", False), log=lambda m: note(job, m))
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
        code, out = kaggle("kernels", "push", "-p", str(folder), "--accelerator", machine)
        for job in ready:
            note(job, (out.splitlines()[-1] if out else f"push exit {code}") + f" (GPU: {GPUS[machine]}"
                 + (f", batch of {len(ready)} reels)" if len(ready) > 1 else ")"))
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
    while True:
        status = kaggle_status(ref)
        settled = time.time() - pushed > 900  # right after a push, Kaggle can still report the previous run
        if status == "queued":
            waited = (time.time() - pushed) / 60
            stage = f"Waiting for a free Kaggle GPU ({GPUS[machine]}, {waited:.0f} min)"
            if machine == "NvidiaTeslaT4" and waited >= GPU_WAIT_MINUTES:
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
            shutil.rmtree(out_dir, ignore_errors=True)
            kaggle("kernels", "output", ref, "-p", str(out_dir), "-o", timeout=900)
            if not (out_dir / ready[0]["id"]).exists() and not settled:
                for job in ready:
                    note(job, "Kaggle still shows the previous run, waiting...")
                time.sleep(30)
                continue
            for job in ready:
                if not job.get("stopped"):
                    collect(job, out_dir / job["id"], status, out_dir / "batch_log.txt")
            return
        if time.time() - started > 6 * 3600:
            for job in ready:
                fail(job, RuntimeError("Kaggle took more than 6 hours; giving up."))
            return
        time.sleep(30)


def collect(job, res_dir, status, batch_log):
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


def worker_loop():
    while True:
        jobs = [load(work_queue.get())]
        time.sleep(3)  # a moment for more topics to arrive, so they can share the run
        while len(jobs) < MAX_BATCH and not work_queue.empty():
            jobs.append(load(work_queue.get()))
        try:
            run_batch(jobs)
        except Exception as e:
            for job in jobs:
                if load(job["id"]).get("status") not in ("done", "error", "stopped"):
                    fail(job, e)


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


def new_job(topic, pdfs):
    jid = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    jdir = JOBS / jid
    (jdir / "uploads").mkdir(parents=True)
    for k, data in enumerate(pdfs):
        (jdir / "uploads" / f"{k:02d}.pdf").write_bytes(data)
    links = [l.strip() for l in request.form.get("links", "").splitlines() if l.strip().startswith("http")]
    job = {"id": jid, "topic": topic, "news": request.form.get("mode") == "news",
           "only_sources": request.form.get("only_sources") == "on", "links": links,
           "press_photos": request.form.get("press_photos") == "on",
           "hook_idea": request.form.get("hook_idea", "").strip()[:300],
           "format": request.form.get("format", "9:16") if request.form.get("format") in ("9:16", "1:1", "16:9") else "9:16",
           "voice": re.sub(r"[^\w-]", "", request.form.get("voice", ""))[:60],
           "music": request.form.get("music", "off") if request.form.get("music") in MUSIC_MOODS else "off",
           "instructions": request.form.get("instructions", "").strip()[:1500],
           "pasted": request.form.get("pasted", ""), "pdf_count": len(list((jdir / "uploads").glob("*.pdf"))),
           "created": time.time(), "status": "queued", "stage": "Waiting in line", "log": []}
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
    job.update(id=new_id, parent_id=jid, forced_hook=forced, hook_options=options if forced else None,
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


def resume_unfinished():
    """After a restart, jobs that never finished are marked so you can retry them."""
    for p in JOBS.glob("*/job.json"):
        j = json.loads(p.read_text(encoding="utf-8"))
        if j.get("status") in ("queued", "working"):
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
    print("Reel Studio is running: open http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
