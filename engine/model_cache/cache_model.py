# One-time Kaggle run (CPU only): download the LLM and keep it as this run's output, so the reel worker
# can attach it as an input instead of downloading 17 GB every time.
import os, subprocess, time

MODEL = "gemma3:27b"
OUT = "/kaggle/working/models"
t0 = time.time()
subprocess.run("apt-get -qq update > /dev/null 2>&1; apt-get -qq install -y zstd > /dev/null 2>&1", shell=True)
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1", shell=True, check=True)
env = {**os.environ, "OLLAMA_MODELS": OUT}
server = subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)
t = time.time()
subprocess.run(["ollama", "pull", MODEL], env=env, check=True, stdout=subprocess.DEVNULL)
print(f"pulled {MODEL} in {time.time() - t:.0f}s", flush=True)
subprocess.run(["ollama", "list"], env=env)
server.terminate()
size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
print(f"cache size {size / 1e9:.1f} GB, total {time.time() - t0:.0f}s", flush=True)
