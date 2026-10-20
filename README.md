# Reel Studio

A free, local website that turns a topic into a short narrated video reel (Reels, Shorts or YouTube).

You type a topic. Research happens on your own computer. A free Kaggle GPU then writes the script with an
open model (Gemma 3), picks photos and clips, adds a voice, captions and music, and sends the finished reel
back to the website so you can watch and download it.

## What it does

- **Research:** every reel starts from recent news (Bing News), plus Wikipedia and your own PDFs, links or
  pasted text. Tick "Only use my sources" to skip the web search.
- **Hooks:** the model writes 15 opening lines, ranks the best 3 and fact-checks the one it uses. Each
  reel shows its top 3 hooks with a **Remake with this hook** button.
- **Script:** a short, clear story of 7 lines, checked for clarity and facts, with each line tagged with
  its source.
- **Photos and clips:** the story's own photos first: every photo inside the news articles (if press photos are
  turned on, even with "Only use my sources", where other articles are read for photos only) and Wikipedia
  photos of the people and places named in the script. Then free sources (Wikimedia Commons, Openverse, NASA,
  and optionally Flickr, Pexels and Pixabay). The model looks at every photo and rejects wrong, fake-looking or
  wrong-event ones; a photo source that is down is skipped for the rest of the reel.
- **Video:** 9:16, 1:1 or 16:9, 10 voices (via edge-tts), optional background music by mood (Kevin MacLeod
  tracks from Wikimedia Commons).
- **Library:** watch, download, remake, stop or delete reels, one at a time or several together.
- **Batches:** type several topics, one per line, and they share one Kaggle run.

## How it works

```
your computer                         Kaggle (free GPU)
-------------                         -----------------
website (Flask)  --->  research  ---> Ollama + Gemma 3: hooks, script, fact check
                                      photo search + checking, voice, ffmpeg video
library  <--------------------------  reel.mp4, script, photo credits
```

The website sends each batch to a private Kaggle notebook (`<your-username>/reel-studio-worker`) and
downloads the result when it finishes. A run usually takes about 10 to 13 minutes once Kaggle gives it a GPU
(a second reel in the same batch adds about 4 to 6). Waiting in Kaggle's queue does not use your GPU hours.
The site always waits for the T4 x2 GPUs (full `gemma3:27b` model). To fall back to a P100 with the smaller
`gemma3:12b` instead, set `GPU_WAIT_MINUTES` in `app.py` to a number of minutes.
If the website is closed or restarted while a reel is on Kaggle, it picks that run up again when it starts.

## Setup (Windows)

1. Install Python 3.11 or newer.
2. In this folder, create the environment and install the requirements:

   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```

3. Make a free Kaggle account, verify your phone number (needed for GPUs and internet in notebooks),
   then sign in once from a terminal in this folder:

   ```
   .venv\Scripts\kaggle.exe auth login
   ```

4. Optional: copy `config.example.json` to `config.json`, or add keys on the website's Settings page.
   All keys are free:
   - Flickr (Creative Commons photos)
   - Pexels and Pixabay (stock video clips)

5. Start the site by double-clicking `start.bat`, or run:

   ```
   .venv\Scripts\python.exe app.py
   ```

   Then open http://127.0.0.1:5050.

## Files

| Path | What it is |
|---|---|
| `app.py` | The website and the background worker that sends batches to Kaggle |
| `templates/index.html` | The whole web page (Create, Library, Settings) |
| `engine/research.py` | News, Wikipedia, PDF and link research, run on your computer |
| `engine/write_script.py` | Hooks, ranking, script, clarity and fact checks |
| `engine/make_reel_fast.py` | Photo and clip search, photo checking, voice, music and video |
| `engine/kaggle_worker_template.py` | The script that runs on Kaggle (setup, model, one reel after another) |
| `engine/scripts/examples.json` | Example scripts that show the writer the style to aim for |
| `engine/model_cache/` | An experiment to cache the model on Kaggle (measured as slower, so it is off) |

## Notes

- Everything is free, but Kaggle limits GPU time (about 30 hours a week) and GPUs are sometimes queued.
- Press photos from news articles are off by default. They are usually copyrighted, so only turn them on
  if you may use them.
- Check the sources listed under each reel before posting it.
- `config.json`, your reels (`jobs/`) and the Python environment (`.venv/`) are not part of this repository.
