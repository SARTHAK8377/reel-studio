"""Post finished reels to Instagram through Meta's official API (Instagram account linked to a Facebook Page).

The video is uploaded straight from this laptop (Meta's "resumable upload"), so it never needs a public link.
Runs on this laptop and uses no GPU.

One-time setup (done by you on Meta's websites): a Creator/Business Instagram account linked to a Facebook Page,
a free Meta developer app, and a user token from the Graph API Explorer. connect() swaps that short token for a
Page token that does not expire.
"""
import time
from pathlib import Path

import requests

VERSION = "v27.0"   # Meta keeps each version working for about 2 years
GRAPH = f"https://graph.facebook.com/{VERSION}"
RUPLOAD = f"https://rupload.facebook.com/ig-api-upload/{VERSION}"
NEEDED = ("instagram_basic", "instagram_content_publish", "pages_show_list", "pages_read_engagement",
          "business_management")
MAX_BYTES = 300 * 1024 * 1024      # Instagram's limit for a reel
PROCESS_WAIT = 15 * 60             # how long to wait for Instagram to process an uploaded video


class IGError(Exception):
    """retry=True: worth trying again later (internet trouble, Instagram busy). False: needs you to fix something."""
    def __init__(self, message, retry=False):
        super().__init__(message)
        self.retry = retry


def _call(method, url, *, params=None, headers=None, data=None, timeout=60):
    try:
        r = requests.request(method, url, params=params, headers=headers, data=data, timeout=timeout)
    except requests.RequestException as e:
        raise IGError(f"No answer from Instagram ({type(e).__name__}). Is the internet on?", retry=True)
    try:
        body = r.json()
    except ValueError:
        body = {}
    err = body.get("error") if isinstance(body, dict) else None
    if r.status_code >= 400 or err:
        err = err or body.get("debug_info") or {}
        msg = err.get("error_user_msg") or err.get("message") or f"HTTP {r.status_code}"
        code = err.get("code")
        if code == 190:
            raise IGError("The Instagram key is no longer valid. Connect again in Settings. (" + msg + ")")
        if code in (10, 200) or "permission" in msg.lower():
            raise IGError("Instagram refused permission: " + msg + ". When you make the token, tick all 5 "
                          "permissions and choose your Page and Instagram account.")
        if code == 9 or err.get("error_subcode") == 2207042:
            raise IGError("Instagram's daily posting limit is reached. It will try again later.", retry=True)
        retry = r.status_code >= 500 or code in (1, 2, 4, 17, 32, 341) or bool(err.get("is_transient")) \
            or bool(err.get("retriable"))
        raise IGError("Instagram said: " + msg, retry=retry)
    return body


def connect(user_token, app_id, app_secret):
    """Turn a Graph API Explorer user token into a never-expiring Page token, and find the linked Instagram account.
    Returns {"token", "ig_id", "username", "page"}."""
    long = _call("GET", f"{GRAPH}/oauth/access_token", params={
        "grant_type": "fb_exchange_token", "client_id": app_id, "client_secret": app_secret,
        "fb_exchange_token": user_token}).get("access_token")
    if not long:
        raise IGError("Meta did not return a long-lasting key. Check the App ID and App Secret.")
    perms = _call("GET", f"{GRAPH}/me/permissions", params={"access_token": long}).get("data", [])
    granted = {p.get("permission") for p in perms if p.get("status") == "granted"}
    missing = [p for p in NEEDED if p not in granted]
    if missing:
        raise IGError("The token is missing these permissions: " + ", ".join(missing) +
                      ". Add them in the Graph API Explorer and generate the token again.")
    pages = _call("GET", f"{GRAPH}/me/accounts", params={
        "fields": "name,access_token,instagram_business_account{id,username}", "access_token": long}).get("data", [])
    for p in pages:
        ig = p.get("instagram_business_account")
        if ig and p.get("access_token"):
            return {"token": p["access_token"], "ig_id": ig["id"], "username": ig.get("username", ""),
                    "page": p.get("name", "")}
    if not pages:
        raise IGError("No Facebook Page was shared with the app. Generate the token again and tick your Page.")
    raise IGError("Your Facebook Page is not linked to an Instagram Creator or Business account. Link it in the "
                  "Instagram app (Edit profile > Page), then connect again.")


def account(cfg):
    """Check the saved key still works. Returns the username."""
    return _call("GET", f"{GRAPH}/{cfg['ig_id']}", params={"fields": "username",
                                                            "access_token": cfg["token"]}).get("username", "")


def posts_left(cfg):
    """(used, allowed) API posts in the last 24 hours, or None if Instagram doesn't say."""
    try:
        d = _call("GET", f"{GRAPH}/{cfg['ig_id']}/content_publishing_limit",
                  params={"fields": "quota_usage,config", "access_token": cfg["token"]}).get("data") or [{}]
        return d[0].get("quota_usage", 0), (d[0].get("config") or {}).get("quota_total", 50)
    except IGError:
        return None


def container_status(cfg, container):
    return _call("GET", f"{GRAPH}/{container}", params={"fields": "status_code,status",
                                                        "access_token": cfg["token"]})


def publish(cfg, video, caption, container=None, on_step=print, on_container=None, ai=True):
    """Upload and publish one reel. container: one made by an earlier try (picked up instead of uploading again,
    so a reel is never posted twice). on_container(id) is called as soon as a container exists, so it can be saved.
    ai: add Instagram's "AI info" label (is_ai_generated, Meta's self-disclosure for AI-made content).
    Returns {"media_id", "permalink"}."""
    video = Path(video)
    size = video.stat().st_size
    if size > MAX_BYTES:
        raise IGError("The video is larger than Instagram's 300 MB limit.")
    token, ig = cfg["token"], cfg["ig_id"]

    state = None
    if container:  # an earlier try got this far: see what happened to it
        try:
            state = container_status(cfg, container).get("status_code")
        except IGError as e:
            if e.retry:
                raise
            state = None
        if state == "PUBLISHED":
            on_step("Already posted by an earlier try")
            return {"media_id": "", "permalink": ""}
        if state not in ("FINISHED", "IN_PROGRESS"):
            container, state = None, None

    if not container:
        on_step("Sending the video to Instagram")
        made = _call("POST", f"{GRAPH}/{ig}/media", params={
            "media_type": "REELS", "upload_type": "resumable", "caption": caption[:2200],
            "share_to_feed": "true", "is_ai_generated": "true" if ai else "false", "access_token": token})
        container = made["id"]
        with open(video, "rb") as f:
            _call("POST", f"{RUPLOAD}/{container}", headers={
                "Authorization": f"OAuth {token}", "offset": "0", "file_size": str(size)}, data=f, timeout=600)
        if on_container:  # saved only once the upload is complete: a failed upload starts fresh next time
            on_container(container)

    on_step("Instagram is processing the video")
    end = time.time() + PROCESS_WAIT
    while state != "FINISHED":
        st = container_status(cfg, container)
        state = st.get("status_code")
        if state == "FINISHED":
            break
        if state == "PUBLISHED":
            return {"media_id": "", "permalink": ""}
        if state in ("ERROR", "EXPIRED"):
            raise IGError("Instagram could not process the video: " + (st.get("status") or state))
        if time.time() > end:
            raise IGError("Instagram took too long to process the video.", retry=True)
        time.sleep(10)

    on_step("Publishing")
    media = _call("POST", f"{GRAPH}/{ig}/media_publish", params={"creation_id": container,
                                                                 "access_token": token})["id"]
    link = ""
    try:
        link = _call("GET", f"{GRAPH}/{media}", params={"fields": "permalink",
                                                        "access_token": token}).get("permalink", "")
    except IGError:
        pass
    return {"media_id": media, "permalink": link}
