# Reel Studio worker: runs on Kaggle (GPU) for a batch of reel jobs, then exits.
# Setup (tools + the LLM) happens once per batch, then each job gets its own folder of results:
#   /kaggle/working/<job id>/ reel.mp4, script.json, photo_credits.txt, result.json, log.txt
# The website fills in JOBS, KEYS and FILES below and pushes this script with the Kaggle API.
import base64, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

JOBS = json.loads(base64.b64decode("__JOBS__").decode("utf-8"))
KEYS = json.loads(base64.b64decode("__KEYS__").decode("utf-8"))   # optional free API keys (Flickr, Pexels, Pixabay)
FILES = json.loads(base64.b64decode("__FILES__").decode("utf-8"))

WORK = Path("/kaggle/working")
ENGINE = WORK / "engine"
BATCH_LOG = open(WORK / "batch_log.txt", "w", encoding="utf-8")
T0 = time.time()
setup_timings = {}
job_log = None


def log(*parts):
    msg = " ".join(str(p) for p in parts)
    print(msg, flush=True)
    BATCH_LOG.write(msg + "\n"); BATCH_LOG.flush()
    if job_log:
        job_log.write(msg + "\n"); job_log.flush()


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if out:
        log(out[-2000:])
    if check and r.returncode:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r


def timed(name, fn):
    t = time.time()
    out = fn()
    setup_timings[name] = round(time.time() - t)
    log(f"setup: {name} took {setup_timings[name]}s")
    return out


def find_model_cache():
    """The cache run's output, if attached: a folder with Ollama's 'manifests' and 'blobs' under /kaggle/input."""
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    for manifests in root.glob("**/manifests"):
        blobs = manifests.parent / "blobs"
        if not blobs.is_dir():
            continue
        # only trust the cache if every file is complete (same size as the model's manifest says)
        for m in [p for p in manifests.rglob("*") if p.is_file()]:
            try:
                layers = json.loads(m.read_text())
                layers = layers.get("layers", []) + [layers.get("config", {})]
            except (ValueError, OSError):
                continue
            bad = [l for l in layers if l.get("digest") and
                   not ((blobs / l["digest"].replace(":", "-")).exists() and
                        (blobs / l["digest"].replace(":", "-")).stat().st_size == l.get("size"))]
            if bad:
                log(f"model cache at {manifests.parent} is incomplete ({len(bad)} file(s) wrong size), not using it")
                return None
        return manifests.parent
    log("no model cache attached")
    return None


def warm_up(model):
    """Load the model onto the GPU with a tiny request (same settings as the writer, so it isn't reloaded)."""
    import urllib.request
    body = json.dumps({"model": model, "prompt": "Say OK.", "stream": False,
                       "options": {"num_ctx": 12288, "num_predict": 2}}).encode()
    urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/generate", body,
                                                  {"Content-Type": "application/json"}), timeout=900).read()


def setup():
    for rel, text in FILES.items():
        p = ENGINE / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    timed("apt (zstd)", lambda: sh("apt-get -qq update > /dev/null 2>&1; apt-get -qq install -y zstd > /dev/null 2>&1",
                                   check=False))
    timed("pip (voice + video tools)", lambda: sh(f"{sys.executable} -m pip -q install edge-tts imageio-ffmpeg"))
    timed("install ollama", lambda: sh("curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1"))
    vram = sum(int(x) for x in subprocess.run("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits",
                                               shell=True, capture_output=True, text=True).stdout.split() or [0])
    model = "gemma3:27b" if vram >= 28000 else "gemma3:12b"
    log(f"GPU memory: {vram} MiB -> model {model}")
    cache = find_model_cache()
    if cache:  # model saved by the one-time cache run: link to it instead of downloading 17 GB
        local = Path("/tmp/ollama_models")
        timed("link cached model", lambda: sh(f"mkdir -p {local} && cp -rs {cache}/. {local}/"))
        os.environ["OLLAMA_MODELS"] = str(local)
        os.environ["OLLAMA_NOPRUNE"] = "1"
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    have = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    if model.split(":")[0] in have and model.split(":")[1] in have:
        log(f"model {model} found in cache at {cache}")
    else:
        if cache:
            log(f"cache at {cache} has no {model}; downloading instead")
        timed("download model", lambda: sh(f"ollama pull {model} > /dev/null 2>&1"))
    timed("model ready (load to GPU)", lambda: warm_up(model))
    timed("font", lambda: sh(f"wget -q -O {ENGINE}/Anton.ttf "
                             "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf", check=False))
    os.environ["REEL_FONT"] = str(ENGINE / "Anton.ttf")
    return model


def make(job, model, out):
    """One reel: script with the local LLM, then the video. Writes everything into `out`."""
    timings = {}
    t = time.time()
    sys.path.insert(0, str(ENGINE))
    import write_script
    script = write_script.write(job["topic"], out / "script.json", job["sources"], job["only_sources"],
                                job["news"], "", model, log=log,
                                instructions=job.get("instructions", ""), hook_idea=job.get("hook_idea", ""),
                                forced_hook=job.get("forced_hook"), forced_options=job.get("hook_options"))
    # research photos and the look/sound settings travel with the script to the reel maker
    script.update(photo_pool=job.get("photo_pool", []), format=job.get("format", "9:16"),
                  voice=job.get("voice", ""), music=job.get("music", "off"))
    (out / "script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"photo pool: {len(script['photo_pool'])}; format {script['format']}; voice {script['voice'] or 'default'}; "
        f"music {script['music']}; extra sources: " + (", ".join(k for k, v in KEYS.items() if v) or "none"))
    timings["script"] = round(time.time() - t)

    t = time.time()
    work = ENGINE / f"out_{job['id']}"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "REEL_VISION_MODEL": model,  # Gemma checks photos and clips
           "FLICKR_API_KEY": KEYS.get("flickr", ""), "PEXELS_API_KEY": KEYS.get("pexels", ""),
           "PIXABAY_API_KEY": KEYS.get("pixabay", "")}
    for attempt in (1, 2):  # a second try reuses the photos, clips and voice already downloaded
        r = subprocess.run([sys.executable, str(ENGINE / "make_reel_fast.py"), str(work),
                            "--script", str(out / "script.json"), "--name", "reel"],
                           capture_output=True, text=True, env=env)
        log((r.stdout + r.stderr)[-12000:])
        if r.returncode == 0:
            break
        log(f"reel maker attempt {attempt} failed" + (", retrying in 20s" if attempt == 1 else ""))
        time.sleep(20)
    else:
        raise RuntimeError("reel maker failed twice (see log)")
    shutil.move(str(work / "reel.mp4"), str(out / "reel.mp4"))
    if (work / "credits.txt").exists():
        shutil.copy(work / "credits.txt", out / "photo_credits.txt")
    shutil.rmtree(work, ignore_errors=True)
    timings["reel"] = round(time.time() - t)
    return timings


def write_result(out, job, status, **extra):
    data = {"job_id": job["id"], "status": status, "setup_timings": setup_timings, **extra}
    (out / "result.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


try:
    log(f"batch of {len(JOBS)} reel(s): " + "; ".join(repr(j["topic"]) for j in JOBS))
    try:
        model = setup()
        setup_error = None
    except Exception as e:
        model, setup_error = None, f"setup failed: {e}"
        log(setup_error); log(traceback.format_exc())
    for n, job in enumerate(JOBS, 1):
        out = WORK / job["id"]
        out.mkdir(parents=True, exist_ok=True)
        job_log = open(out / "log.txt", "w", encoding="utf-8")
        log(f"\n=== reel {n}/{len(JOBS)}: {job['topic']!r} ({'news' if job['news'] else 'general'}, "
            f"{len(job['sources'])} sources)")
        try:
            if setup_error:
                raise RuntimeError(setup_error)
            t = time.time()
            timings = make(job, model, out)
            write_result(out, job, "done", model=model, timings={**timings, "setup": sum(setup_timings.values())},
                         seconds=round(time.time() - t))
            log(f"reel {n} done: {timings}")
        except Exception as e:
            log("ERROR:", e); log(traceback.format_exc())
            write_result(out, job, "error", error=str(e))
        finally:
            job_log.close(); job_log = None
    log(f"\nBATCH DONE in {round(time.time() - T0)}s; setup: {setup_timings}")
finally:
    shutil.rmtree(ENGINE, ignore_errors=True)  # only send back the results, not the tools
    BATCH_LOG.close()
