#!/usr/bin/env python3
"""
Yellow Car Detection Bot with Hugging Face OWLv2 Fallback
This version includes Hugging Face OWLv2 as fallback when GitHub Models API is rate limited.
"""

import base64
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from PIL import Image
from atproto import Client, models
from dotenv import load_dotenv
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

load_dotenv()

TOKEN = os.getenv("KEY_GITHUB_TOKEN")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o"

OWL_V2_MODEL_ID = "google/owlv2-base-patch16-ensemble"

TODAY_FOLDER = Path("today")
TODAY_FOLDER.mkdir(exist_ok=True)
WEBCAM_URLS_FILE = Path("valid_webcam_ids.txt")
SHUFFLE_STATE_FILE = Path("shuffle_state.json")
BSKY_HANDLE = os.getenv("BSKY_HANDLE")
BSKY_PASSWORD = os.getenv("BSKY_PASSWORD")

MAX_RUNTIME_MINUTES = 20
IMAGES_PER_SESSION = 30
YELLOW_THRESHOLD = 150
MIN_CLUSTER_SIZE = 80

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for OWLv2 model (load once)
owl_processor = None
owl_model = None


class RateLimitException(Exception):
    pass


def load_owlv2_model():
    """Load OWLv2 model and processor once at startup"""
    global owl_processor, owl_model
    
    if owl_processor is not None and owl_model is not None:
        return True
        
    if not HF_API_TOKEN:
        logging.error("Hugging Face API token is not defined for OWLv2.")
        return False
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading OWLv2 model on device: {device}")
        
        owl_processor = Owlv2Processor.from_pretrained(OWL_V2_MODEL_ID)
        owl_model = Owlv2ForObjectDetection.from_pretrained(OWL_V2_MODEL_ID).to(device)
        
        logging.info("✅ OWLv2 model loaded successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to load OWLv2 model or processor: {e}")
        return False


def load_shuffle_state():
    if SHUFFLE_STATE_FILE.exists():
        try:
            with open(SHUFFLE_STATE_FILE, 'r') as f:
                state = json.load(f)
                if not isinstance(state.get("shuffled_urls"), list):
                    return {"shuffled_urls": [], "current_index": 0, "stats": {"total_processed": 0, "total_posted": 0}}
                return state
        except Exception as e:
            logging.warning(f"Could not load shuffle state: {e}")
    return {"shuffled_urls": [], "current_index": 0, "stats": {"total_processed": 0, "total_posted": 0}}


def save_shuffle_state(state):
    try:
        with open(SHUFFLE_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logging.error(f"Could not save shuffle state: {e}")


def get_shuffled_urls():
    if not WEBCAM_URLS_FILE.exists():
        logging.error(f"Webcam URLs file not found: {WEBCAM_URLS_FILE}")
        return [], 0, {}

    with open(WEBCAM_URLS_FILE, "r") as f:
        all_urls = [line.strip() for line in f if line.strip()]

    state = load_shuffle_state()

    if not state["shuffled_urls"] or state["current_index"] >= len(state["shuffled_urls"]):
        logging.info(f"Shuffling {len(all_urls)} webcam URLs for fair processing")
        state["shuffled_urls"] = all_urls.copy()
        random.shuffle(state["shuffled_urls"])
        state["current_index"] = 0
        cycle_num = state.get("cycle_count", 0) + 1
        state["cycle_count"] = cycle_num
        save_shuffle_state(state)

    return state["shuffled_urls"], state["current_index"], state.get("stats", {"total_processed": 0, "total_posted": 0})


def update_shuffle_state(new_index, stats_update=None):
    state = load_shuffle_state()
    state["current_index"] = new_index
    if stats_update:
        if "stats" not in state:
            state["stats"] = {"total_processed": 0, "total_posted": 0}
        for key, value in stats_update.items():
            state["stats"][key] = state["stats"].get(key, 0) + value
    save_shuffle_state(state)


def download_image(url, dest, timeout=10):
    try:
        resp = requests.get(url, allow_redirects=True, timeout=timeout)
        if resp.status_code == 200:
            with open(dest, "wb") as f:
                f.write(resp.content)
            return True
        else:
            logging.debug(f"Failed to download {url}: Status {resp.status_code}")
            return False
    except Exception as e:
        logging.debug(f"Exception downloading {url}: {e}")
        return False


def find_yellow_clusters(image_path, min_cluster_size=MIN_CLUSTER_SIZE):
    try:
        img = Image.open(image_path).convert('RGB')
        pixels = img.load()
        width, height = img.size
        yellow_count = 0

        for y in range(0, height, 2):
            for x in range(0, width, 2):
                r, g, b = pixels[x, y]
                if r > YELLOW_THRESHOLD and g > YELLOW_THRESHOLD and b < 100:
                    yellow_count += 1
                    if yellow_count >= min_cluster_size:
                        return True

        return yellow_count >= min_cluster_size
    except Exception as e:
        logging.debug(f"Error processing {image_path}: {e}")
        return False


def get_image_data_url(image_file, image_format):
    try:
        with open(image_file, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        return f"data:image/{image_format};base64,{image_base64}"
    except Exception as e:
        logging.error(f"Could not read '{image_file}': {e}")
        return None


def ask_owlv2_if_yellow_car(image_path):
    """Fixed OWLv2 fallback function that properly handles tensor results"""
    global owl_processor, owl_model
    
    if owl_processor is None or owl_model is None:
        if not load_owlv2_model():
            return None

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Could not load image for OWLv2: {e}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        text_queries = [["a yellow car"]]
        logging.info(f"Querying OWLv2 with text: '{text_queries[0][0]}'")

        inputs = owl_processor(text=text_queries, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = owl_model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])  # (H, W)

        results = owl_processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.1
        )

        # Fixed: Properly check if any detections were made
        if len(results) > 0 and len(results[0]["boxes"]) > 0:
            boxes = results[0]["boxes"]
            scores = results[0]["scores"]
            labels = results[0]["labels"]
            
            # Check if any detection has sufficient confidence
            found_yellow_car = False
            for i, score in enumerate(scores):
                confidence = score.item()  # Convert tensor to Python float
                if confidence > 0.15:  # Confidence threshold
                    logging.info(f"🟡 OWLv2 detected yellow car with confidence: {confidence:.3f}")
                    found_yellow_car = True
                    break
            
            if found_yellow_car:
                logging.info(f"🟡 OWLv2 detected yellow car(s)!")
                return "yes"
            else:
                logging.info(f"🚫 OWLv2 detections below confidence threshold.")
                return "no"
        else:
            logging.info(f"🚫 OWLv2 found no yellow cars.")
            return "no"
            
    except Exception as e:
        logging.error(f"Error in OWLv2 processing: {e}")
        return None


def ask_ai_if_yellow_car(image_path):
    fallback_triggered = False
    if not TOKEN:
        logging.error("Azure API token is not defined")
        fallback_triggered = True
        return ask_owlv2_if_yellow_car(image_path), fallback_triggered

    image_data_url = get_image_data_url(image_path, "jpg")
    if not image_data_url:
        return None, fallback_triggered

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    body = {
        "messages": [
            {
                "role": "user",
                "content": f"Does this image show a yellow car? {image_data_url}"
            }
        ],
        "model": MODEL_NAME,
        "max_tokens": 10
    }

    try:
        resp = requests.post(f"{ENDPOINT}/chat/completions", json=body, headers=headers, timeout=30)

        if resp.status_code == 429:
            quota_remaining = resp.headers.get("x-ms-user-quota-remaining", "unknown")
            quota_resets_after = resp.headers.get("x-ms-user-quota-resets-after", "unknown")

            logging.warning("🚫 Rate limit hit (429) - switching to OWLv2 fallback:")
            logging.warning(f"   Quota remaining: {quota_remaining}")
            logging.warning(f"   Quota resets after: {quota_resets_after}")

            try:
                reset_time = datetime.fromisoformat(quota_resets_after.replace('Z', '+00:00'))
                current_time = datetime.now(reset_time.tzinfo)
                time_until_reset = reset_time - current_time

                if time_until_reset.total_seconds() > 0:
                    minutes = int(time_until_reset.total_seconds() / 60)
                    seconds = int(time_until_reset.total_seconds() % 60)
                    logging.warning(f"   Time until quota reset: {minutes}m {seconds}s")
                else:
                    logging.warning("   Quota should be available now")
            except Exception as e:
                logging.debug(f"Could not parse reset time: {e}")

            logging.info("🔄 Switching to OWLv2 model...")
            fallback_triggered = True
            return ask_owlv2_if_yellow_car(image_path), fallback_triggered

        if resp.status_code != 200:
            logging.error(f"Azure API error: {resp.status_code}")
            logging.debug(f"Response headers: {dict(resp.headers)}")
            logging.info("🔄 Trying OWLv2 fallback due to Azure error...")
            fallback_triggered = True
            return ask_owlv2_if_yellow_car(image_path), fallback_triggered

        data = resp.json()
        result = data["choices"][0]["message"]["content"].strip().lower()
        logging.info(f"✅ Azure AI response: {result}")
        return result, fallback_triggered

    except Exception as e:
        logging.error(f"Error calling Azure AI endpoint: {e}")
        logging.info("🔄 Trying OWLv2 fallback due to Azure error...")
        fallback_triggered = True
        return ask_owlv2_if_yellow_car(image_path), fallback_triggered


def post_to_bluesky(image_path, alt_text):
    if not BSKY_HANDLE or not BSKY_PASSWORD:
        logging.error("Bluesky credentials not defined")
        return False

    try:
        client = Client()
        client.login(BSKY_HANDLE.strip(), BSKY_PASSWORD.strip())

        image_data_url = get_image_data_url(image_path, "jpg")
        if not image_data_url:
            return False

        header, encoded = image_data_url.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        blob = client.upload_blob(image_bytes).blob

        client.app.bsky.feed.post.create(
            repo=client.me.did,
            record=models.AppBskyFeedPost.Record(
                text="GUL BIL!",
                created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                embed=models.AppBskyEmbedImages.Main(
                    images=[
                        models.AppBskyEmbedImages.Image(
                            image=blob,
                            alt=alt_text
                        )
                    ]
                )
            )
        )
        logging.info("Successfully posted yellow car to Bluesky!")
        return True
    except Exception as e:
        logging.error(f"Error posting to Bluesky: {e}")
        return False


def main():
    start_time = datetime.now()
    max_end_time = start_time + timedelta(minutes=MAX_RUNTIME_MINUTES)

    logging.info(f"Starting Yellow Car Bot - will run for max {MAX_RUNTIME_MINUTES} minutes")

    # Pre-load OWLv2 model if HF token is available
    if HF_API_TOKEN:
        if load_owlv2_model():
            logging.info("✅ Hugging Face OWLv2 fallback ready")
        else:
            logging.warning("⚠️  Failed to load OWLv2 - fallback not available")
    else:
        logging.warning("⚠️  No Hugging Face token - fallback not available")

    urls, current_index, current_stats = get_shuffled_urls()
    if not urls:
        logging.error("No URLs available")
        return

    logging.info(f"Resuming from position {current_index}/{len(urls)}")
    logging.info(f"All-time stats: {current_stats.get('total_processed', 0)} processed, {current_stats.get('total_posted', 0)} posted")

    session_processed = 0
    session_yellow_found = 0
    session_posted = 0
    fallback_used = 0
    final_index = 0

    try:
        for i in range(current_index, min(current_index + IMAGES_PER_SESSION, len(urls))):
            if datetime.now() >= max_end_time:
                logging.info("Time limit reached, stopping gracefully")
                break

            url = urls[i]
            timestamp = int(time.time())
            image_name = f"cam_{i + 1}_{timestamp}.jpg"
            image_path = TODAY_FOLDER / image_name

            logging.info(f"Processing {i + 1}/{len(urls)}: downloading image...")

            if not download_image(url, image_path):
                continue

            session_processed += 1

            if find_yellow_clusters(image_path):
                session_yellow_found += 1
                logging.info(f"🟡 Yellow cluster detected! Checking with AI...")

                ai_response, was_fallback_used = ask_ai_if_yellow_car(image_path)
                logging.info(f"AI response: {ai_response}")
                if was_fallback_used:
                    fallback_used += 1

                if ai_response and "yes" in ai_response:
                    logging.info("🚗 YELLOW CAR CONFIRMED! Posting to Bluesky...")
                    if post_to_bluesky(image_path, alt_text="Yellow car spotted on traffic camera!"):
                        session_posted += 1
                        logging.info("✅ Posted to Bluesky successfully!")

            try:
                image_path.unlink()
            except:
                pass

            time.sleep(1)

        final_index = min(current_index + session_processed, len(urls))
        stats_update = {
            "total_processed": session_processed,
            "total_posted": session_posted
        }
        update_shuffle_state(final_index, stats_update)

    except KeyboardInterrupt:
        logging.info("Interrupted, saving progress...")
        final_index = current_index + session_processed
        stats_update = {"total_processed": session_processed, "total_posted": session_posted}
        update_shuffle_state(final_index, stats_update)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    runtime = datetime.now() - start_time
    updated_stats = load_shuffle_state().get("stats", {})

    logging.info(f"\n=== SESSION SUMMARY ===")
    logging.info(f"Runtime: {runtime.total_seconds():.1f} seconds")
    logging.info(f"Images processed: {session_processed}")
    logging.info(f"Yellow clusters found: {session_yellow_found}")
    logging.info(f"Cars posted: {session_posted}")
    if fallback_used > 0:
        logging.info(f"OWLv2 fallbacks used: {fallback_used}")
    logging.info(f"Progress: {final_index}/{len(urls)}")
    logging.info(f"All-time totals: {updated_stats.get('total_processed', 0)} processed, {updated_stats.get('total_posted', 0)} posted")


if __name__ == "__main__":
    main()