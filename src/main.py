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
MIN_CLUSTER_SIZE = 120

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
        
        # Track yellow pixel positions to check for rectangular clustering
        yellow_pixels = []

        for y in range(0, height, 2):
            for x in range(0, width, 2):
                r, g, b = pixels[x, y]
                if r > YELLOW_THRESHOLD and g > YELLOW_THRESHOLD and b < 100:
                    yellow_count += 1
                    yellow_pixels.append((x, y))
                    
                    # Early exit if we have enough pixels
                    if yellow_count >= min_cluster_size:
                        # Additional validation: check if yellow pixels form reasonable clusters
                        # (not just scattered road markings)
                        if len(yellow_pixels) >= min_cluster_size:
                            # Check for clustered distribution (not just thin lines)
                            x_coords = [p[0] for p in yellow_pixels[-min_cluster_size:]]
                            y_coords = [p[1] for p in yellow_pixels[-min_cluster_size:]]
                            
                            x_range = max(x_coords) - min(x_coords)
                            y_range = max(y_coords) - min(y_coords)
                            
                            # Require both width and height (not just thin lines)
                            # Road markings are typically very thin in one dimension
                            if x_range > 20 and y_range > 15:  # Minimum rectangular area
                                logging.debug(f"Yellow cluster found: {yellow_count} pixels, dimensions: {x_range}x{y_range}")
                                return True

        # Final check with all pixels if we didn't early exit
        if yellow_count >= min_cluster_size and len(yellow_pixels) >= min_cluster_size:
            x_coords = [p[0] for p in yellow_pixels]
            y_coords = [p[1] for p in yellow_pixels]
            
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            
            # Require rectangular distribution
            if x_range > 20 and y_range > 15:
                logging.debug(f"Final yellow cluster found: {yellow_count} pixels, dimensions: {x_range}x{y_range}")
                return True
            else:
                logging.debug(f"Yellow pixels too linear: {yellow_count} pixels, dimensions: {x_range}x{y_range}")

        return False
    except Exception as e:
        logging.debug(f"Error processing {image_path}: {e}")
        return False


def get_image_data_url(image_file, image_format, max_size=(800, 600), quality=85):
    """
    Get base64 data URL for image, with automatic resizing to avoid 413 errors
    """
    try:
        # Open and potentially resize image
        with Image.open(image_file) as img:
            img = img.convert('RGB')
            
            # Check if image needs resizing
            original_size = img.size
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                # Resize maintaining aspect ratio
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                logging.debug(f"Resized image from {original_size} to {img.size}")
            
            # Save to bytes with compression
            import io
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=quality, optimize=True)
            img_bytes.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(img_bytes.getvalue()).decode()
            
            # Log final size
            final_size_kb = len(image_base64) * 3 / 4 / 1024  # Approximate KB
            logging.debug(f"Final image payload: {final_size_kb:.1f} KB")
            
        return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        logging.error(f"Could not read/process '{image_file}': {e}")
        return None


def ask_owlv2_if_yellow_car(image_path):
    """Strict OWLv2 fallback function that avoids false positives from road markings"""
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
        # More specific queries to avoid road markings and other yellow objects
        text_queries = [["a yellow car", "a yellow vehicle", "a yellow automobile", "a yellow taxi"]]
        logging.info(f"Querying OWLv2 with multiple car-specific prompts")

        inputs = owl_processor(text=text_queries, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = owl_model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])  # (H, W)

        # Use higher threshold for initial filtering
        results = owl_processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.25  # Increased from 0.1 to 0.25
        )

        # Strict validation: require high confidence AND reasonable bounding box
        if len(results) > 0 and len(results[0]["boxes"]) > 0:
            boxes = results[0]["boxes"]
            scores = results[0]["scores"]
            labels = results[0]["labels"]
            
            image_width, image_height = image.size
            found_yellow_car = False
            
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                confidence = score.item()
                
                # STRICT CONFIDENCE: Require very high confidence
                if confidence < 0.35:  # Increased from 0.15 to 0.35
                    continue
                
                # BOUNDING BOX VALIDATION: Check if box looks like a car
                x1, y1, x2, y2 = box.tolist()
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                image_area = image_width * image_height
                
                # Box should be reasonable size (not tiny road markings, not entire image)
                area_ratio = box_area / image_area
                aspect_ratio = box_width / box_height if box_height > 0 else 0
                
                # Cars typically have aspect ratio between 1.2 and 3.0
                # And should occupy reasonable portion of image (not tiny markings)
                if (area_ratio < 0.005 or  # Too small (likely road marking)
                    area_ratio > 0.8 or    # Too large (likely not a car)
                    aspect_ratio < 0.8 or  # Too tall (not car-like)
                    aspect_ratio > 4.0):   # Too wide (likely road marking)
                    logging.debug(f"Rejected detection: area_ratio={area_ratio:.4f}, aspect_ratio={aspect_ratio:.2f}")
                    continue
                
                # POSITION VALIDATION: Cars are usually not at very bottom (road markings area)
                bottom_y_ratio = y2 / image_height
                if bottom_y_ratio > 0.95:  # Very bottom of image (likely road marking)
                    logging.debug(f"Rejected detection: too close to bottom edge")
                    continue
                
                # If we get here, it's a high-confidence, well-positioned, car-shaped detection
                label_text = text_queries[0][label.item()]
                logging.info(f"🟡 HIGH-CONFIDENCE yellow car detected!")
                logging.info(f"   Label: '{label_text}'")
                logging.info(f"   Confidence: {confidence:.3f}")
                logging.info(f"   Box area ratio: {area_ratio:.4f}")
                logging.info(f"   Aspect ratio: {aspect_ratio:.2f}")
                logging.info(f"   Position: ({x1:.0f},{y1:.0f}) to ({x2:.0f},{y2:.0f})")
                
                found_yellow_car = True
                break
            
            if found_yellow_car:
                logging.info(f"🟡 OWLv2 CONFIRMED yellow car with strict validation!")
                return "yes"
            else:
                logging.info(f"🚫 OWLv2 detections failed strict validation (likely road markings/false positives)")
                return "no"
        else:
            logging.info(f"🚫 OWLv2 found no yellow cars above threshold.")
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

    image_data_url = get_image_data_url(image_path, "jpg", max_size=(800, 600), quality=85)
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

        if resp.status_code == 413:
            logging.warning("🚫 Payload too large (413) - trying with smaller image...")
            # Try with much smaller image
            smaller_image_url = get_image_data_url(image_path, "jpg", max_size=(400, 300), quality=70)
            if smaller_image_url:
                body["messages"][0]["content"] = f"Does this image show a yellow car? {smaller_image_url}"
                resp = requests.post(f"{ENDPOINT}/chat/completions", json=body, headers=headers, timeout=30)
                
                if resp.status_code == 413:
                    logging.error("Image still too large even after aggressive compression - switching to OWLv2")
                    fallback_triggered = True
                    return ask_owlv2_if_yellow_car(image_path), fallback_triggered
                elif resp.status_code != 200:
                    logging.error(f"Azure API error after resize: {resp.status_code}")
                    fallback_triggered = True
                    return ask_owlv2_if_yellow_car(image_path), fallback_triggered
                else:
                    # Success with smaller image
                    data = resp.json()
                    result = data["choices"][0]["message"]["content"].strip().lower()
                    logging.info(f"✅ Azure AI response (resized): {result}")
                    return result, fallback_triggered
            else:
                logging.error("Could not create smaller image - switching to OWLv2")
                fallback_triggered = True
                return ask_owlv2_if_yellow_car(image_path), fallback_triggered

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

        image_data_url = get_image_data_url(image_path, "jpg", max_size=(400, 300), quality=70)
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