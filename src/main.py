#!/usr/bin/env python3
"""
Yellow Car Detection Bot with Facebook AI Fallback
This version includes Hugging Face DETR models as fallback when GitHub Models API is rate limited.
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

load_dotenv()

TOKEN = os.getenv("KEY_GITHUB_TOKEN")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o"

# Hugging Face fallback models
HF_MODEL_URLS = [
    "https://api-inference.huggingface.co/models/facebook/detr-resnet-50",
    "https://api-inference.huggingface.co/models/facebook/detr-resnet-101",
]

# Color classification models for yellow detection
HF_COLOR_MODELS = [
    "https://api-inference.huggingface.co/models/google/mobilenet_v2_1.0_224",
    "https://api-inference.huggingface.co/models/microsoft/resnet-50",
]

TODAY_FOLDER = Path("today")
TODAY_FOLDER.mkdir(exist_ok=True)
WEBCAM_URLS_FILE = Path("valid_webcam_ids.txt")
SHUFFLE_STATE_FILE = Path("shuffle_state.json")
BSKY_HANDLE = os.getenv("BSKY_HANDLE")
BSKY_PASSWORD = os.getenv("BSKY_PASSWORD")

# Configuration - Optimized for 21 minutes max runtime
MAX_RUNTIME_MINUTES = 20  # Stop at 20 minutes to be safe
IMAGES_PER_SESSION = 30  # Process ~30 images per run (can adjust based on performance)
YELLOW_THRESHOLD = 150  # Lower threshold for yellow detection (more sensitive)
MIN_CLUSTER_SIZE = 80  # Smaller cluster size for detection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RateLimitException(Exception):
    """Custom exception for rate limiting"""
    pass


def load_shuffle_state():
    """Load the current shuffle state from file"""
    if SHUFFLE_STATE_FILE.exists():
        try:
            with open(SHUFFLE_STATE_FILE, 'r') as f:
                state = json.load(f)
                # Validate state structure
                if not isinstance(state.get("shuffled_urls"), list):
                    return {"shuffled_urls": [], "current_index": 0, "stats": {"total_processed": 0, "total_posted": 0}}
                return state
        except Exception as e:
            logging.warning(f"Could not load shuffle state: {e}")
    return {"shuffled_urls": [], "current_index": 0, "stats": {"total_processed": 0, "total_posted": 0}}


def save_shuffle_state(state):
    """Save the current shuffle state to file"""
    try:
        with open(SHUFFLE_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logging.error(f"Could not save shuffle state: {e}")


def get_shuffled_urls():
    """Get URLs in shuffled order, reshuffling when list is exhausted"""
    if not WEBCAM_URLS_FILE.exists():
        logging.error(f"Webcam URLs file not found: {WEBCAM_URLS_FILE}")
        return [], 0, {}

    with open(WEBCAM_URLS_FILE, "r") as f:
        all_urls = [line.strip() for line in f if line.strip()]

    state = load_shuffle_state()

    # If we need to reshuffle (first run or list exhausted)
    if not state["shuffled_urls"] or state["current_index"] >= len(state["shuffled_urls"]):
        logging.info(f"Shuffling {len(all_urls)} webcam URLs for fair processing")
        state["shuffled_urls"] = all_urls.copy()
        random.shuffle(state["shuffled_urls"])
        state["current_index"] = 0
        cycle_num = state.get("cycle_count", 0) + 1
        state["cycle_count"] = cycle_num
        logging.info(f"Starting cycle #{cycle_num}")
        save_shuffle_state(state)

    return state["shuffled_urls"], state["current_index"], state.get("stats", {"total_processed": 0, "total_posted": 0})


def update_shuffle_state(new_index, stats_update=None):
    """Update the current index and stats in shuffle state"""
    state = load_shuffle_state()
    state["current_index"] = new_index
    if stats_update:
        if "stats" not in state:
            state["stats"] = {"total_processed": 0, "total_posted": 0}
        for key, value in stats_update.items():
            state["stats"][key] = state["stats"].get(key, 0) + value
    save_shuffle_state(state)


def download_image(url, dest, timeout=10):
    """Download image with shorter timeout for efficiency"""
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
    """Optimized yellow detection"""
    try:
        img = Image.open(image_path).convert('RGB')
        pixels = img.load()
        width, height = img.size
        yellow_count = 0

        # Sample every 2nd pixel for speed (still accurate for cars)
        for y in range(0, height, 2):
            for x in range(0, width, 2):
                r, g, b = pixels[x, y]
                # More permissive yellow detection
                if r > YELLOW_THRESHOLD and g > YELLOW_THRESHOLD and b < 100:
                    yellow_count += 1
                    # Early exit if we have enough yellow pixels
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


def check_for_yellow_in_classification(classifications):
    """Check if classification results suggest yellow objects or vehicles"""
    yellow_related_terms = [
        'yellow', 'golden', 'amber', 'lemon', 'canary', 'school bus', 
        'taxi', 'cab', 'banana', 'gold', 'mustard', 'sunshine'
    ]
    
    car_related_terms = [
        'car', 'vehicle', 'automobile', 'sedan', 'coupe', 'hatchback',
        'wagon', 'suv', 'taxi', 'cab', 'bus', 'van', 'truck'
    ]
    
    for item in classifications:
        label = item.get('label', '').lower()
        score = item.get('score', 0)
        
        # Look for yellow-related terms with decent confidence
        has_yellow = any(term in label for term in yellow_related_terms)
        has_vehicle = any(term in label for term in car_related_terms)
        
        if has_yellow and score > 0.1:
            logging.info(f"Found yellow-related classification: {label} (confidence: {score:.3f})")
            if has_vehicle or score > 0.3:
                return True
        
        # Special case for taxi/cab which are often yellow
        if ('taxi' in label or 'cab' in label) and score > 0.2:
            logging.info(f"Found taxi/cab classification: {label} (confidence: {score:.3f})")
            return True
    
    return False


def ask_color_classification_model(image_path):
    """Use general image classification models to detect yellow cars"""
    if not HF_API_TOKEN:
        return None
    
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        logging.error(f"Could not read image file: {e}")
        return None
    
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
    }
    
    # Try color classification models
    for model_url in HF_COLOR_MODELS:
        model_name = model_url.split('/')[-1]
        logging.info(f"Trying color classification model: {model_name}")
        
        try:
            response = requests.post(model_url, headers=headers, data=image_bytes, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logging.debug(f"Classification response: {result}")
                
                has_yellow = check_for_yellow_in_classification(result)
                if has_yellow:
                    logging.info(f"✅ Color classification ({model_name}) detected yellow vehicle-related content!")
                    return "yes"
                else:
                    logging.info(f"❌ Color classification ({model_name}) found no yellow vehicle content")
                    continue
                    
            elif response.status_code == 503:
                logging.warning(f"Model {model_name} is loading, trying next model...")
                continue
            else:
                logging.warning(f"Model {model_name} returned {response.status_code}, trying next...")
                continue
                
        except Exception as e:
            logging.warning(f"Error with color model {model_name}: {e}")
            continue
    
    return "no"
    """Check if DETR detected vehicle objects with strict filtering to prevent false positives"""
    vehicle_labels = {'car', 'truck', 'bus', 'motorcycle', 'van', 'vehicle'}
    
    vehicles_found = []
    for obj in detections:
        label = obj.get("label", "").lower()
        score = obj.get("score", 0)
        
        # Look for vehicle-related labels
        if any(v in label for v in vehicle_labels):
            vehicles_found.append({
                'label': label,
                'score': score,
                'obj': obj
            })
    
    if not vehicles_found:
        return False
    
    # Sort by confidence score 
    vehicles_found.sort(key=lambda x: x['score'], reverse=True)
    
    # Much stricter requirements to prevent false positives:
    # 1. Higher confidence threshold (0.7 instead of 0.3)
    # 2. Must specifically be "car" (not truck/bus which are less likely to be yellow)
    # 3. Multiple vehicles increase confidence
    
    car_detections = [v for v in vehicles_found if 'car' in v['label'] and v['score'] > 0.7]
    
    if car_detections:
        best_car = car_detections[0]
        logging.info(f"DETR detected {best_car['label']} with HIGH confidence {best_car['score']:.3f}")
        return True
    
    # Also check for multiple lower-confidence vehicles (could indicate real traffic)
    high_conf_vehicles = [v for v in vehicles_found if v['score'] > 0.5]
    if len(high_conf_vehicles) >= 2:
        logging.info(f"DETR detected {len(high_conf_vehicles)} vehicles with moderate confidence")
        return True
    
    # Log what we found but rejected
    if vehicles_found:
        best = vehicles_found[0]
        logging.info(f"DETR found {best['label']} but confidence too low: {best['score']:.3f} (need >0.7 for cars)")
    
    return False


def ask_facebook_ai_if_yellow_car(image_path):
    """Enhanced fallback using both vehicle detection and color classification"""
    if not HF_API_TOKEN:
        logging.error("Hugging Face API token is not defined")
        return None
    
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        logging.error(f"Could not read image file: {e}")
        return None
    
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "image/jpeg"
    }
    
    # Step 1: Check if there are vehicles using DETR models
    vehicles_detected = False
    for model_url in HF_MODEL_URLS:
        model_name = model_url.split('/')[-1]
        logging.info(f"Checking for vehicles with DETR model: {model_name}")
        
        try:
            response = requests.post(model_url, headers=headers, data=image_bytes, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logging.debug(f"DETR response: {result}")
                
                has_car = check_for_car_in_detr_response(result)
                if has_car:
                    logging.info(f"✅ DETR ({model_name}) detected HIGH-CONFIDENCE vehicle!")
                    vehicles_detected = True
                    break  # Found vehicles, proceed to color check
                else:
                    logging.info(f"❌ DETR ({model_name}) found no high-confidence vehicles")
                    continue
                    
            elif response.status_code == 503:
                logging.warning(f"DETR model {model_name} is loading, trying next...")
                continue
            else:
                logging.warning(f"DETR model {model_name} returned {response.status_code}, trying next...")
                continue
                
        except Exception as e:
            logging.warning(f"Error with DETR model {model_name}: {e}")
            continue
    
    # Step 2: If vehicles detected, check for yellow color
    if vehicles_detected:
        logging.info("🔍 Vehicles found! Now checking for yellow color...")
        
        # Try color classification models
        for model_url in HF_COLOR_MODELS:
            model_name = model_url.split('/')[-1]
            logging.info(f"Checking for yellow with classification model: {model_name}")
            
            try:
                response = requests.post(model_url, headers=headers, data=image_bytes, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    logging.debug(f"Classification response: {result}")
                    
                    has_yellow = check_for_yellow_in_classification(result)
                    if has_yellow:
                        logging.info(f"🟡 YELLOW CAR CONFIRMED by multi-model approach!")
                        logging.info(f"   Vehicle detection: ✅ (DETR)")
                        logging.info(f"   Yellow detection: ✅ ({model_name})")
                        return "yes"
                        
                elif response.status_code == 503:
                    logging.warning(f"Classification model {model_name} is loading, trying next...")
                    continue
                else:
                    logging.warning(f"Classification model {model_name} returned {response.status_code}")
                    continue
                    
            except Exception as e:
                logging.warning(f"Error with classification model {model_name}: {e}")
                continue
        
        # If we have vehicles but no yellow detected
        logging.info("🚗 Vehicles detected but no strong yellow indicators found")
        return "no"
    
    else:
        # Step 3: If no vehicles detected by DETR, try direct color classification
        # (might catch cases where DETR missed vehicles but there are yellow cars)
        logging.info("🔍 No vehicles detected by DETR, trying direct color classification...")
        color_result = ask_color_classification_model(image_path)
        if color_result == "yes":
            logging.info("🟡 Direct color classification suggests yellow vehicle content!")
            return "yes"
    
    # If we get here, no yellow cars detected by any method
    logging.info("🚫 No yellow cars detected by Facebook AI fallback models")
    return "no"


def ask_ai_if_yellow_car(image_path):
    """Primary AI query with Facebook AI fallback on rate limiting"""
    if not TOKEN:
        logging.error("Azure API token is not defined")
        # Try fallback immediately if no primary token
        return ask_facebook_ai_if_yellow_car(image_path)

    image_data_url = get_image_data_url(image_path, "jpg")
    if not image_data_url:
        return None

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    body = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer with only 'yes' or 'no'."},
            {"role": "user", "content": [
                {"type": "text",
                 "text": "Is there a yellow car visible in this traffic camera image? Answer only 'yes' or 'no'."},
                {"type": "image_url", "image_url": {"url": image_data_url, "detail": "low"}}
            ]}
        ],
        "model": MODEL_NAME,
        "max_tokens": 10  # Limit response length
    }

    try:
        resp = requests.post(f"{ENDPOINT}/chat/completions", json=body, headers=headers, timeout=30)

        if resp.status_code == 429:
            # Log detailed rate limit information
            quota_remaining = resp.headers.get("x-ms-user-quota-remaining", "unknown")
            quota_resets_after = resp.headers.get("x-ms-user-quota-resets-after", "unknown")

            logging.warning("🚫 Rate limit hit (429) - switching to Facebook AI fallback:")
            logging.warning(f"   Quota remaining: {quota_remaining}")
            logging.warning(f"   Quota resets after: {quota_resets_after}")

            # Try to parse the reset time for a more user-friendly message
            if quota_resets_after != "unknown":
                try:
                    from datetime import datetime
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

            # Use Facebook AI fallback instead of stopping
            logging.info("🔄 Switching to Facebook DETR models...")
            return ask_facebook_ai_if_yellow_car(image_path)

        if resp.status_code != 200:
            logging.error(f"Azure API error: {resp.status_code}")
            # Log response headers for debugging other errors too
            if resp.headers:
                logging.debug(f"Response headers: {dict(resp.headers)}")
            # Try fallback for other errors too
            logging.info("🔄 Trying Facebook AI fallback due to Azure error...")
            return ask_facebook_ai_if_yellow_car(image_path)

        data = resp.json()
        result = data["choices"][0]["message"]["content"].strip().lower()
        logging.info(f"✅ Azure AI response: {result}")
        return result

    except Exception as e:
        logging.error(f"Error calling Azure AI endpoint: {e}")
        logging.info("🔄 Trying Facebook AI fallback due to Azure error...")
        return ask_facebook_ai_if_yellow_car(image_path)


def post_to_bluesky(image_path, alt_text):
    """Streamlined Bluesky posting"""
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
                            alt=alt_text,
                            image=blob
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
    
    # Check if we have fallback credentials
    if HF_API_TOKEN:
        logging.info("✅ Hugging Face fallback available")
    else:
        logging.warning("⚠️  No Hugging Face token - fallback not available")

    # Get shuffled URLs and current position
    urls, current_index, current_stats = get_shuffled_urls()
    if not urls:
        logging.error("No URLs available")
        return

    logging.info(
        f"Resuming from position {current_index}/{len(urls)} (cycle progress: {current_index / len(urls) * 100:.1f}%)")
    logging.info(
        f"All-time stats: {current_stats.get('total_processed', 0)} processed, {current_stats.get('total_posted', 0)} posted")

    session_processed = 0
    session_yellow_found = 0
    session_posted = 0
    fallback_used = 0
    final_index = 0  # Ensure final_index is always defined
    
    try:
        for i in range(current_index, min(current_index + IMAGES_PER_SESSION, len(urls))):
            # Check time limit
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

                ai_response = ask_ai_if_yellow_car(image_path)
                logging.info(f"AI response: {ai_response}")

                if ai_response and "yes" in ai_response:
                    logging.info("🚗 YELLOW CAR CONFIRMED! Posting to Bluesky...")
                    if post_to_bluesky(image_path, alt_text="Yellow car spotted on traffic camera!"):
                        session_posted += 1
                        logging.info("✅ Posted to Bluesky successfully!")

            # Clean up image to save space
            try:
                image_path.unlink()
            except:
                pass

            # Brief pause to avoid overwhelming APIs
            time.sleep(1)

        # Update state with final position
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

    # Final summary
    runtime = datetime.now() - start_time
    updated_stats = load_shuffle_state().get("stats", {})

    logging.info(f"\n=== SESSION SUMMARY ===")
    logging.info(f"Runtime: {runtime.total_seconds():.1f} seconds ({runtime.total_seconds() / 60:.1f} minutes)")
    logging.info(f"Images processed this session: {session_processed}")
    logging.info(f"Yellow clusters found: {session_yellow_found}")
    logging.info(f"Cars posted to Bluesky: {session_posted}")
    if fallback_used > 0:
        logging.info(f"Facebook AI fallbacks used: {fallback_used}")
    logging.info(f"Progress: {final_index}/{len(urls)} ({final_index / len(urls) * 100:.1f}% of current cycle)")
    logging.info(
        f"All-time totals: {updated_stats.get('total_processed', 0)} processed, {updated_stats.get('total_posted', 0)} posted")

    if session_yellow_found > 0:
        logging.info(f"Yellow detection rate: {session_yellow_found / session_processed * 100:.1f}%")
        if session_posted > 0:
            logging.info(f"Confirmation rate: {session_posted / session_yellow_found * 100:.1f}%")


if __name__ == "__main__":
    main()