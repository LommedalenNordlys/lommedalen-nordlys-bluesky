#!/usr/bin/env python3
"""
Lommedalen Aurora Borealis Detection Bot for Bluesky
Monitors webcams for northern lights and posts to Bluesky when detected.
"""

import base64
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import requests
from PIL import Image
from dotenv import load_dotenv

# Optional Bluesky client import (if you use it)
try:
    from atproto import Client, models
except Exception:
    Client = None
    models = None

load_dotenv()

# Config (set these in your environment or .env file)
# NOTE: We used to capture TOKEN at import time, but tests set the env var after importing.
# To support dynamic updates (e.g. setting KEY_GITHUB_TOKEN in test cases), we now resolve it
# on demand via get_ai_token(). The legacy TOKEN variable remains for backward compatibility
# but may be None and is no longer authoritative.
TOKEN = os.getenv("KEY_GITHUB_TOKEN") or os.getenv("AZURE_API_KEY") or os.getenv("API_KEY")  # deprecated static snapshot

def get_ai_token() -> Optional[str]:
    """Return current AI auth token from environment.

    Checks several possible variable names to allow flexibility in CI/local setups:
    - KEY_GITHUB_TOKEN (used in tests)
    - AZURE_API_KEY (potential Azure key name)
    - API_KEY (generic fallback)

    Returns:
        The token string if found, else None.
    """
    return (
        os.getenv("KEY_GITHUB_TOKEN")
        or os.getenv("AZURE_API_KEY")
        or os.getenv("API_KEY")
        or TOKEN  # fallback to legacy snapshot if still populated
    )
ENDPOINT = os.getenv("AZURE_ENDPOINT") or "https://models.inference.ai.azure.com"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o"

TODAY_FOLDER = Path("today")
TODAY_FOLDER.mkdir(exist_ok=True)
WEBCAM_LOCATIONS_FILE = Path("webcam_locations.json")
SUN_SCHEDULE_FILE = Path("sun_schedule.json")
SHUFFLE_STATE_FILE = Path("shuffle_state.json")
BSKY_HANDLE = os.getenv("BSKY_HANDLE")
BSKY_PASSWORD = os.getenv("BSKY_PASSWORD")

MAX_RUNTIME_MINUTES = int(os.getenv("MAX_RUNTIME_MINUTES", "20"))
IMAGES_PER_SESSION = int(os.getenv("IMAGES_PER_SESSION", "30"))
MIN_KP_INDEX = float(os.getenv("MIN_KP", "1"))  # Minimum planetary Kp index required to proceed (lowered to 1 for broader capture)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
azure_rate_limited = False

# Aurora / Kp data source (Ovation model latest snapshot)
OVATION_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"

def fetch_current_kp(lat: float = 59.0, lon: float = 10.0) -> Optional[float]:
    """Fetch a local Kp-like value for given coordinates (default Lommedalen area ~59N,10E) using Ovation model.

    The endpoint returns a JSON object containing a list of coordinate triples: [latitude, longitude, value].
    Example target triple: [59, 10, 0] -> we interpret the third number as localized activity (proxy for Kp / auroral intensity).

    Return:
        float value (third element) if exact or nearest coordinate found, else None.
    Notes:
        - This is a localized intensity proxy, not necessarily the global planetary Kp.
        - We perform nearest-neighbor match (within 1 degree) if exact integer lat/lon not present.
    """
    try:
        resp = requests.get(OVATION_URL, timeout=12)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logging.warning(f"Ovation request failed: {e}")
        return None

    coords = data.get("coordinates") or data.get("data") or []
    if not isinstance(coords, list) or not coords:
        logging.debug("Ovation payload missing 'coordinates' list")
        return None

    target_lat = int(round(lat))
    target_lon = int(round(lon))

    exact_match_val: Optional[float] = None
    nearest_val: Optional[float] = None
    nearest_dist: float = 9999.0

    for triple in coords:
        if not (isinstance(triple, list) and len(triple) >= 3):
            continue
        c_lat, c_lon, c_val = triple[0], triple[1], triple[2]
        # Basic validation of numeric types
        if not (isinstance(c_lat, (int, float)) and isinstance(c_lon, (int, float)) and isinstance(c_val, (int, float))):
            continue
        if int(c_lat) == target_lat and int(c_lon) == target_lon:
            exact_match_val = float(c_val)
            break
        # Track nearest within reasonable range (<=1 degree separation)
        dist = abs(c_lat - lat) + abs(c_lon - lon)
        if dist < nearest_dist and dist <= 1.0:
            nearest_dist = dist
            nearest_val = float(c_val)

    kp_local = exact_match_val if exact_match_val is not None else nearest_val
    if kp_local is None:
        logging.debug("No suitable Ovation coordinate match for specified location")
        return None

    logging.info(f"Local Ovation intensity (proxy Kp) at ~({target_lat},{target_lon}) = {kp_local:.2f}")
    return kp_local

def load_cached_sun_times(file: Path = SUN_SCHEDULE_FILE) -> Optional[Dict[str, Any]]:
    """Load cached sun & twilight times from sun_schedule.json written by the update workflow.

    Returns a dict with datetime objects for keys needed by darkness logic, or None if unavailable.
    File format example (truncated):
    {
      "updated_at": "2025-11-01T10:33:46.622676Z",
      "sun_times": {
         "astronomical_twilight_begin": "2025-11-01T04:18:25+00:00",
         ...
      }
    }
    """
    if not file.exists():
        logging.debug(f"Sun schedule file missing: {file}")
        return None
    try:
        with open(file, "r") as f:
            data = json.load(f)
        sun_times = data.get("sun_times") or {}
        if not isinstance(sun_times, dict):
            return None
        parsed: Dict[str, Any] = {}
        for k, v in sun_times.items():
            if not isinstance(v, str):
                continue
            try:
                # Accept API ISO strings with or without trailing Z
                iso = v.replace("Z", "+00:00")
                parsed[k] = datetime.fromisoformat(iso)
            except Exception:
                logging.debug(f"Could not parse cached sun time {k}: {v}")
        return parsed if parsed else None
    except Exception as e:
        logging.warning(f"Failed to load cached sun times: {e}")
        return None

def is_dark(parsed_times: Dict[str, Any]) -> bool:
    """Determine if it's sufficiently dark for aurora detection.

    Strategy:
    - Prefer astronomical darkness: current UTC time >= astronomical_twilight_end OR < astronomical_twilight_begin.
    - If astronomical keys missing, fallback to nautical twilight end/begin.
    - If still missing, fallback to civil twilight end/begin.
    - If data unavailable, assume it's potentially dark (return True) to avoid skipping when uncertain.
    """
    now = datetime.now(timezone.utc)

    # Helper to evaluate darkness window (outside of begin..end interval)
    def outside(begin_key: str, end_key: str) -> Optional[bool]:
        start = parsed_times.get(begin_key)
        end = parsed_times.get(end_key)
        if start and end:
            # Dark if current time < start OR > end (outside twilight interval)
            return now < start or now > end
        return None

    # Priority order
    for begin, end in [
        ("astronomical_twilight_begin", "astronomical_twilight_end"),
        ("nautical_twilight_begin", "nautical_twilight_end"),
        ("civil_twilight_begin", "civil_twilight_end"),
    ]:
        result = outside(begin, end)
        if result is not None:
            return result

    # If no usable data, assume darkness to proceed
    logging.warning("Twilight boundary data incomplete; assuming dark to proceed.")
    return True

def describe_light_condition(parsed_times: Dict[str, Any]) -> str:
    """Produce a human-readable string about current light condition based on twilight phases."""
    now = datetime.now(timezone.utc)
    at_begin = parsed_times.get("astronomical_twilight_begin")
    at_end = parsed_times.get("astronomical_twilight_end")
    if at_begin and at_end:
        if at_begin <= now <= at_end:
            return "Astronomical twilight (not fully dark)"
    nt_begin = parsed_times.get("nautical_twilight_begin")
    nt_end = parsed_times.get("nautical_twilight_end")
    if nt_begin and nt_end and nt_begin <= now <= nt_end:
        return "Nautical twilight"
    ct_begin = parsed_times.get("civil_twilight_begin")
    ct_end = parsed_times.get("civil_twilight_end")
    if ct_begin and ct_end and ct_begin <= now <= ct_end:
        return "Civil twilight (still fairly bright)"
    return "Dark"


def load_shuffle_state() -> Dict[str, Any]:
    if SHUFFLE_STATE_FILE.exists():
        try:
            with open(SHUFFLE_STATE_FILE, 'r') as f:
                state = json.load(f)
                if not isinstance(state.get("shuffled_urls", []), list):
                    return {"shuffled_urls": [], "current_index": 0, "stats": {"total_processed": 0, "total_posted": 0}}
                return state
        except Exception as e:
            logging.warning(f"Could not load shuffle state: {e}")
    return {"shuffled_urls": [], "current_index": 0, "stats": {"total_processed": 0, "total_posted": 0}}


def save_shuffle_state(state: Dict[str, Any]) -> None:
    try:
        with open(SHUFFLE_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logging.error(f"Could not save shuffle state: {e}")


def load_webcam_locations() -> List[Dict[str, str]]:
    """
    Expected webcam_locations.json format:
    {
      "webcams": [
        {"place": "Tryvann", "urls": ["https://example.com/cam1.jpg", "https://example.com/cam1_live.jpg"]},
        {"place": "Lommedalen", "urls": ["https://example.com/lom1.jpg"]}
      ]
    }
    Returns a flat list of {"url": <url>, "place": <place>}
    """
    if not WEBCAM_LOCATIONS_FILE.exists():
        logging.error(f"Webcam locations file not found: {WEBCAM_LOCATIONS_FILE}")
        return []

    try:
        with open(WEBCAM_LOCATIONS_FILE, "r") as f:
            data = json.load(f)
        webcam_list = []
        for item in data.get("webcams", []):
            place = item.get("place", "Unknown")
            for url in item.get("urls", []):
                webcam_list.append({"url": url, "place": place})
        return webcam_list
    except Exception as e:
        logging.error(f"Error loading webcam locations: {e}")
        return []


def get_shuffled_urls() -> Tuple[List[Dict[str, str]], int, Dict[str, int]]:
    webcam_list = load_webcam_locations()
    if not webcam_list:
        return [], 0, {"total_processed": 0, "total_posted": 0}

    state = load_shuffle_state()

    if not state.get("shuffled_urls") or state.get("current_index", 0) >= len(state["shuffled_urls"]):
        logging.info(f"Shuffling {len(webcam_list)} webcams for fair processing")
        state["shuffled_urls"] = webcam_list.copy()
        random.shuffle(state["shuffled_urls"])
        state["current_index"] = 0
        state["cycle_count"] = state.get("cycle_count", 0) + 1
        save_shuffle_state(state)

    return state["shuffled_urls"], state.get("current_index", 0), state.get("stats", {"total_processed": 0, "total_posted": 0})


def update_shuffle_state(new_index: int, stats_update: Optional[Dict[str, int]] = None) -> None:
    state = load_shuffle_state()
    state["current_index"] = new_index
    if stats_update:
        if "stats" not in state:
            state["stats"] = {"total_processed": 0, "total_posted": 0}
        for key, value in stats_update.items():
            state["stats"][key] = state["stats"].get(key, 0) + value
    save_shuffle_state(state)


def download_image(url: str, dest: Path, timeout: int = 20) -> bool:
    """Download image from url -> dest. Returns True on success."""
    try:
        headers = {"User-Agent": "AuroraBot/1.0"}
        r = requests.get(url, headers=headers, stream=True, timeout=timeout)
        if r.status_code != 200:
            logging.warning(f"Failed to download image (status {r.status_code}): {url}")
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1024):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        logging.debug(f"Download error for {url}: {e}")
        return False


def check_for_aurora_colors(image_path: Path, min_green_pixels: int = 50) -> bool:
    """
    Quick pre-check for aurora-like colors (green/purple in dark sky).
    Returns True if the image likely contains aurora colors.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        pixels = img.load()
        width, height = img.size
        green_count = 0
        purple_count = 0

        # sample pixels for performance
        step = max(1, min(width, height) // 150)  # adaptive sampling
        for y in range(0, height, step):
            for x in range(0, width, step):
                r, g, b = pixels[x, y]
                # simple heuristics for green aurora
                if g > 100 and g > r and g > b and (r < 150 or b < 150):
                    green_count += 1
                # purple/pink aurora heuristic
                elif r > 100 and b > 80 and g < r and g < b:
                    purple_count += 1
                if green_count + purple_count >= min_green_pixels:
                    logging.debug(f"Aurora-like colors found: green={green_count}, purple={purple_count}")
                    return True
        return (green_count + purple_count) >= min_green_pixels
    except Exception as e:
        logging.debug(f"Error processing {image_path}: {e}")
        return False


def get_image_data_url(image_file: Path, max_size=(800, 600), quality: int = 85) -> Optional[str]:
    """Return a base64 data URL for the image (JPEG) after optional resizing."""
    try:
        img = Image.open(image_file)
        original = img.size
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            logging.debug(f"Resized image from {original} to {img.size}")
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logging.error(f"Could not create data URL for {image_file}: {e}")
        return None


def ask_ai_if_aurora(image_path: Path, location: str) -> Optional[str]:
    """
    Ask an external AI endpoint whether the image contains aurora.
    Returns 'yes'/'no' or None on error.
    This implementation is intentionally generic — adjust payload for your specific model endpoint.
    """
    global azure_rate_limited

    if azure_rate_limited:
        logging.info("Azure previously rate-limited; skipping ask_ai_if_aurora")
        return None

    token = get_ai_token()
    if not token:
        logging.error("No API token configured for AI calls")
        return None

    # Use smaller image size to avoid token limits (max 8000 tokens for gpt-4o)
    image_data_url = get_image_data_url(image_path, max_size=(400, 300), quality=60)
    if not image_data_url:
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    # Minimal prompt asking for yes/no answer
    prompt_text = (
        "Does this webcam image show aurora borealis (northern lights) in the sky? "
        "Look for green, purple, or pink glowing lights in the night sky. Answer with only 'yes' or 'no'."
    )

    # Azure OpenAI-compatible format
    body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 10
    }

    try:
        resp = requests.post(f"{ENDPOINT}/chat/completions", json=body, headers=headers, timeout=30)
    except Exception as e:
        logging.error(f"Error calling AI endpoint: {e}")
        return None

    # If image is still too large, retry with even smaller size
    if resp.status_code == 413:
        logging.warning("Image still too large (413), retrying with smaller size...")
        smaller_image_url = get_image_data_url(image_path, max_size=(200, 150), quality=50)
        if smaller_image_url:
            body["messages"][0]["content"][1]["image_url"]["url"] = smaller_image_url
            try:
                resp = requests.post(f"{ENDPOINT}/chat/completions", json=body, headers=headers, timeout=30)
            except Exception as e:
                logging.error(f"Retry failed: {e}")
                return None
        else:
            logging.error("Could not create smaller image")
            return None

    if resp.status_code == 429:
        azure_rate_limited = True
        logging.warning("Azure rate-limited (429). Will skip further Azure calls this session.")
        return None
    if resp.status_code != 200:
        logging.error(f"AI endpoint returned status {resp.status_code}: {resp.text[:400]}")
        return None

    try:
        data = resp.json()
        # Try a few common response locations depending on provider
        candidate = None
        # Common OpenAI-style
        if "choices" in data and data["choices"]:
            candidate = data["choices"][0].get("message", {}).get("content") or data["choices"][0].get("text")
        # Some endpoints return 'output' or 'result'
        if not candidate:
            candidate = data.get("output") or data.get("result") or data.get("content")
        if not candidate:
            logging.debug("AI returned no textual content")
            return None
        result = str(candidate).strip().lower()
        logging.info(f"AI response: {result}")
        if "yes" in result:
            return "yes"
        if "no" in result:
            return "no"
        # fallback: return raw
        return result
    except Exception as e:
        logging.error(f"Error parsing AI response: {e}")
        return None


def build_post_text(location: str, kp_value: Optional[float], test: bool) -> str:
    """Return formatted Bluesky post text with optional test disclaimer.

    Args:
        location: Location string.
        kp_value: Optional Kp/local intensity value.
        test: Whether this is a test post (adds disclaimer).
    """
    kp_part = f"Kp index: {kp_value:.1f}" if kp_value is not None else "Kp index: n/a"
    base_text = f"🌌 Nordlys / Aurora Borealis!\n📍 {location}\n{kp_part}"
    if test:
        return (
            "🧪 Test Aurora Detection (System Accuracy Check)\n"
            f"{base_text}\n(This is a test post; validating multi-source KP + image analysis pipeline)"
        )
    return base_text

def post_to_bluesky(image_path: Path, location: str, kp_value: Optional[float] = None, test: bool = False) -> bool:
    """Post aurora sighting to Bluesky with location and KP index.

    Args:
        image_path: Path to image to upload.
        location: Human-readable location string.
        kp_value: Optional Kp / local intensity value; if provided will be included in the post text.
    """
    if Client is None or models is None:
        logging.error("Bluesky atproto client not available (install 'atproto' or adjust code).")
        return False
    if not BSKY_HANDLE or not BSKY_PASSWORD:
        logging.error("Bluesky credentials not defined")
        return False

    try:
        client = Client()
        client.login(BSKY_HANDLE.strip(), BSKY_PASSWORD.strip())

        image_data_url = get_image_data_url(image_path, max_size=(800, 600), quality=85)
        if not image_data_url:
            return False

        header, encoded = image_data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        # Upload returns an object with .blob in original code; adapt if different in your client
        upload_result = client.upload_blob(image_bytes)
        blob_ref = getattr(upload_result, "blob", upload_result)

        post_text = build_post_text(location, kp_value, test)

        client.app.bsky.feed.post.create(
            repo=client.me.did,
            record=models.AppBskyFeedPost.Record(
                text=post_text,
                created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                embed=models.AppBskyEmbedImages.Main(
                    images=[
                        models.AppBskyEmbedImages.Image(
                            image=blob_ref,
                            alt=f"Aurora borealis spotted at {location}"
                        )
                    ]
                )
            )
        )
        logging.info(f"Successfully posted aurora from {location} to Bluesky!")
        return True
    except Exception as e:
        logging.error(f"Error posting to Bluesky: {e}")
        return False


def main():
    start_time = datetime.now()
    max_end_time = start_time + timedelta(minutes=MAX_RUNTIME_MINUTES)

    logging.info(f"Starting Aurora Borealis Detection Bot - will run for max {MAX_RUNTIME_MINUTES} minutes")

    # Darkness check using cached schedule (fast, no API call)
    sun_times = load_cached_sun_times()
    if sun_times:
        light_desc = describe_light_condition(sun_times)
        dark = is_dark(sun_times)
        logging.info(f"Light condition: {light_desc}; dark={dark}")
        if not dark:
            logging.info("☀️  It is not dark enough for aurora detection. Exiting early to conserve resources.")
            return
    else:
        logging.warning("Could not retrieve sun times; proceeding without darkness gate.")

    # Note: KP index validation is handled by bash scripts (check_kp_multi.sh)
    # before this Python script runs in the GitHub Actions workflow.
    # No need to re-validate here.

    global azure_rate_limited
    azure_rate_limited = False

    webcams, current_index, stats = get_shuffled_urls()
    if not webcams:
        logging.error("No webcams found; exiting")
        return

    logging.info(f"Resuming from position {current_index}/{len(webcams)}")
    logging.info(f"All-time stats: processed={stats.get('total_processed', 0)} posted={stats.get('total_posted', 0)}")

    if not get_ai_token():
        logging.error("No AI token configured — AI checks will fail")
    else:
        logging.info("AI token present — will attempt AI verification (subject to rate limits)")

    session_processed = 0
    session_aurora_colors_found = 0
    session_posted = 0

    try:
        for i in range(current_index, min(current_index + IMAGES_PER_SESSION, len(webcams))):
            if datetime.now() >= max_end_time:
                logging.info("Time limit reached, stopping")
                break

            cam = webcams[i]
            url = cam["url"]
            location = cam.get("place", "Unknown")
            image_name = f"cam_{i + 1}_{int(time.time())}.jpg"
            image_path = TODAY_FOLDER / image_name

            logging.info(f"Processing {i + 1}/{len(webcams)} - {location}: {url}")
            if not download_image(url, image_path):
                logging.info("  ❌ Failed to download image")
                continue

            session_processed += 1

            if check_for_aurora_colors(image_path):
                session_aurora_colors_found += 1
                logging.info("  🌟 Aurora-like colors detected — asking AI for confirmation")
                ai_answer = ask_ai_if_aurora(image_path, location)
                if ai_answer == "yes":
                    logging.info("  🌌 AI confirms aurora — posting to Bluesky...")
                    if post_to_bluesky(image_path, location, kp_val):
                        session_posted += 1
                        logging.info("  ✅ Posted to Bluesky")
                elif ai_answer == "no":
                    logging.info("  ❌ AI denies aurora (false positive)")
                else:
                    logging.info("  ⚠️ Could not verify with AI (skipped/failure)")
            else:
                logging.debug("  No aurora-like colors")

            # cleanup downloaded image
            try:
                image_path.unlink()
            except Exception:
                pass

            time.sleep(1.5)

        final_index = min(current_index + session_processed, len(webcams))
        update_shuffle_state(final_index, {"total_processed": session_processed, "total_posted": session_posted})

    except KeyboardInterrupt:
        logging.info("Interrupted; saving progress")
        final_index = current_index + session_processed
        update_shuffle_state(final_index, {"total_processed": session_processed, "total_posted": session_posted})
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")

    runtime = datetime.now() - start_time
    updated_stats = load_shuffle_state().get("stats", {})
    logging.info("=== SESSION SUMMARY ===")
    logging.info(f"Runtime: {runtime.total_seconds():.1f}s")
    logging.info(f"Images processed: {session_processed}")
    logging.info(f"Aurora-like colors found: {session_aurora_colors_found}")
    logging.info(f"Aurora posted: {session_posted}")
    logging.info(f"Progress: {min(current_index + session_processed, len(webcams))}/{len(webcams)}")
    logging.info(f"All-time totals: processed={updated_stats.get('total_processed', 0)}, posted={updated_stats.get('total_posted', 0)}")


if __name__ == "__main__":
    main()
