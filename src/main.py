"""
Yellow Car Detection Bot (REST HuggingFace API BLIP version)
Detects yellow vehicles using 'Salesforce/blip-image-captioning-base'
"""

import base64
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image
from atproto import Client, models
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
class Config:
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    BSKY_HANDLE = os.getenv("BSKY_HANDLE")
    BSKY_PASSWORD = os.getenv("BSKY_PASSWORD")
    TODAY_FOLDER = Path("today")
    WEBCAM_URLS_FILE = Path("valid_webcam_ids.txt")
    SHUFFLE_STATE_FILE = Path("shuffle_state.json")

    MAX_RUNTIME_MINUTES = 20
    IMAGES_PER_SESSION = 30

    YELLOW_THRESHOLD = 150
    MIN_CLUSTER_SIZE = 80
    DOWNLOAD_TIMEOUT = 10
    API_TIMEOUT = 45

    REQUEST_DELAY = 1.0  # seconds

    @classmethod
    def validate(cls) -> bool:
        missing = []
        if not cls.HF_API_TOKEN:
            missing.append("HF_API_TOKEN")
        if not cls.BSKY_HANDLE:
            missing.append("BSKY_HANDLE")
        if not cls.BSKY_PASSWORD:
            missing.append("BSKY_PASSWORD")
        if not cls.WEBCAM_URLS_FILE.exists():
            missing.append(f"Webcam URLs file: {cls.WEBCAM_URLS_FILE}")
        if missing:
            logging.error(f"Missing required configuration: {', '.join(missing)}")
            return False
        return True

# Ensure directories exist
Config.TODAY_FOLDER.mkdir(exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Config.TODAY_FOLDER / 'bot.log')
    ]
)

class RateLimitException(Exception):
    pass

class ShuffleStateManager:
    @staticmethod
    def load_state() -> Dict:
        default_state = {
            "shuffled_urls": [],
            "current_index": 0,
            "cycle_count": 0,
            "stats": {"total_processed": 0, "total_posted": 0}
        }
        if not Config.SHUFFLE_STATE_FILE.exists():
            return default_state
        try:
            with open(Config.SHUFFLE_STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            for key, value in default_state.items():
                if key not in state:
                    state[key] = value
            return state
        except (json.JSONDecodeError, IOError):
            return default_state

    @staticmethod
    def save_state(state: Dict) -> None:
        try:
            with open(Config.SHUFFLE_STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except IOError as e:
            logging.error(f"Could not save shuffle state: {e}")

    @staticmethod
    def get_shuffled_urls() -> Tuple[List[str], int, Dict]:
        try:
            with open(Config.WEBCAM_URLS_FILE, "r", encoding='utf-8') as f:
                all_urls = [line.strip() for line in f if line.strip()]
        except IOError as e:
            logging.error(f"Could not read webcam URLs file: {e}")
            return [], 0, {}
        state = ShuffleStateManager.load_state()
        if (not state["shuffled_urls"] or
            state["current_index"] >= len(state["shuffled_urls"]) or
            len(state["shuffled_urls"]) != len(all_urls)):
            logging.info(f"Shuffling {len(all_urls)} webcam URLs for fair processing")
            state["shuffled_urls"] = all_urls.copy()
            random.shuffle(state["shuffled_urls"])
            state["current_index"] = 0
            state["cycle_count"] = state.get("cycle_count", 0) + 1
            ShuffleStateManager.save_state(state)
        return state["shuffled_urls"], state["current_index"], state["stats"]

    @staticmethod
    def update_progress(new_index: int, stats_update: Optional[Dict] = None) -> None:
        state = ShuffleStateManager.load_state()
        state["current_index"] = new_index
        if stats_update:
            for key, value in stats_update.items():
                state["stats"][key] = state["stats"].get(key, 0) + value
        ShuffleStateManager.save_state(state)

class ImageProcessor:
    @staticmethod
    def download_image(url: str, dest: Path) -> bool:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(
                url,
                headers=headers,
                allow_redirects=True,
                timeout=Config.DOWNLOAD_TIMEOUT,
                stream=True
            )
            if resp.status_code == 200:
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                return False
        except requests.RequestException:
            return False

    @staticmethod
    def find_yellow_clusters(image_path: Path) -> bool:
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                width, height = img.size
                grid_size = 20
                yellow_regions = []
                for y in range(0, height - grid_size, grid_size):
                    for x in range(0, width - grid_size, grid_size):
                        yellow_count = 0
                        total_pixels = 0
                        for dy in range(0, grid_size, 2):
                            for dx in range(0, grid_size, 2):
                                px_x, px_y = x + dx, y + dy
                                if px_x < width and px_y < height:
                                    r, g, b = img.getpixel((px_x, px_y))
                                    total_pixels += 1
                                    if (r > Config.YELLOW_THRESHOLD and
                                        g > Config.YELLOW_THRESHOLD and
                                        b < 120 and
                                        r + g > 2 * b):
                                        yellow_count += 1
                        if total_pixels > 0:
                            yellow_percentage = (yellow_count / total_pixels) * 100
                            if yellow_percentage > 25 and yellow_count > 10:
                                yellow_regions.append({'yellow_pixels': yellow_count})
                if not yellow_regions:
                    return False
                total_yellow_pixels = sum(region['yellow_pixels'] for region in yellow_regions)
                substantial_regions = [r for r in yellow_regions if r['yellow_pixels'] > 20]
                return (len(substantial_regions) >= 1 or
                        (len(yellow_regions) >= 2 and total_yellow_pixels > Config.MIN_CLUSTER_SIZE))
        except Exception:
            return False

class AIDetector:
    """
    AI detection using Hugging Face's API-enabled image captioning model:
    'Salesforce/blip-image-captioning-base'
    """
    @staticmethod
    def detect_yellow_vehicle(image_path: Path) -> Optional[str]:
        model_name = "Salesforce/blip-image-captioning-base"
        endpoint = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {
            "Authorization": f"Bearer {Config.HF_API_TOKEN}",
        }
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
        except Exception as e:
            logging.error(f"Cannot read image: {e}")
            return None
        try:
            response = requests.post(endpoint, headers=headers, data=image_data, timeout=Config.API_TIMEOUT)
        except Exception as e:
            logging.error(f"Request to HF API failed: {e}")
            return None

        if response.status_code == 429:
            logging.warning("Rate limit hit for HuggingFace model")
            raise RateLimitException()
        if response.status_code == 503:
            logging.info(f"Model {model_name} is loading.")
            return None
        if response.status_code == 404:
            logging.error(f"Model {model_name} not found (404).")
            return None
        if response.status_code == 401:
            logging.error("Unauthorized (401) - check HF_API_TOKEN.")
            return None
        if response.status_code != 200:
            logging.error(f"HF API error {response.status_code}: {response.text}")
            return None

        try:
            result = response.json()
            logging.info(f"HuggingFace API result: {result}")
            if isinstance(result, list) and result and 'generated_text' in result[0]:
                caption = result[0]['generated_text'].lower().strip()
                return AIDetector._analyze_caption(caption)
            elif isinstance(result, dict) and 'generated_text' in result:
                caption = result['generated_text'].lower().strip()
                return AIDetector._analyze_caption(caption)
            else:
                return None
        except Exception as e:
            logging.error(f"Error parsing HF API result: {e}")
            return None

    @staticmethod
    def _analyze_caption(caption: str) -> str:
        vehicle_keywords = [
            'car', 'truck', 'van', 'bus', 'vehicle', 'taxi', 'automobile',
            'suv', 'sedan', 'coupe', 'hatchback', 'convertible'
        ]
        yellow_keywords = ['yellow', 'gold', 'golden', 'bright yellow', 'lemon']
        yellow_vehicle_phrases = [
            'yellow car', 'yellow truck', 'yellow van', 'yellow bus',
            'yellow taxi', 'taxi cab', 'yellow vehicle', 'gold car',
            'golden car', 'bright yellow car', 'yellow sedan', 'yellow suv'
        ]
        has_yellow_vehicle_phrase = any(phrase in caption for phrase in yellow_vehicle_phrases)
        if has_yellow_vehicle_phrase:
            return "yes"
        has_vehicle = any(keyword in caption for keyword in vehicle_keywords)
        has_yellow = any(keyword in caption for keyword in yellow_keywords)
        if has_vehicle and has_yellow:
            if len(caption.split()) <= 12:
                return "yes"
            else:
                return "maybe"
        elif has_vehicle:
            return "vehicle_found_no_yellow"
        else:
            return "no"

class BlueskyPoster:
    @staticmethod
    def post_image(image_path: Path, alt_text: str) -> bool:
        try:
            client = Client()
            client.login(Config.BSKY_HANDLE.strip(), Config.BSKY_PASSWORD.strip())
            with open(image_path, "rb") as f:
                image_data = f.read()
            blob = client.upload_blob(image_data).blob
            post_text = "GUL BIL! 🟡🚗"
            client.app.bsky.feed.post.create(
                repo=client.me.did,
                record=models.AppBskyFeedPost.Record(
                    text=post_text,
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

class YellowCarBot:
    def __init__(self):
        self.start_time = datetime.now()
        self.max_end_time = self.start_time + timedelta(minutes=Config.MAX_RUNTIME_MINUTES)
        self.session_stats = {
            'processed': 0,
            'yellow_clusters': 0,
            'ai_confirmations': 0,
            'posted': 0
        }

    def cleanup_image(self, image_path: Path) -> None:
        try:
            if image_path.exists():
                image_path.unlink()
        except OSError:
            pass

    def process_single_image(self, url: str, index: int) -> bool:
        timestamp = int(time.time())
        image_name = f"cam_{index + 1}_{timestamp}.jpg"
        image_path = Config.TODAY_FOLDER / image_name
        try:
            logging.info(f"Processing {index + 1}: downloading image...")
            if not ImageProcessor.download_image(url, image_path):
                return False
            self.session_stats['processed'] += 1
            if ImageProcessor.find_yellow_clusters(image_path):
                self.session_stats['yellow_clusters'] += 1
                logging.info("🟡 Yellow cluster detected! Checking with AI...")
                try:
                    ai_response = AIDetector.detect_yellow_vehicle(image_path)
                    logging.info(f"AI response: {ai_response}")
                    if ai_response == "yes":
                        self.session_stats['ai_confirmations'] += 1
                        logging.info("🚗 YELLOW CAR CONFIRMED! Posting to Bluesky...")
                        if BlueskyPoster.post_image(
                            image_path,
                            alt_text="Yellow car spotted on traffic camera!"
                        ):
                            self.session_stats['posted'] += 1
                            logging.info("✅ Posted to Bluesky successfully!")
                            return True
                except RateLimitException:
                    logging.warning("Rate limit reached: stopping run.")
                    raise
            return False
        finally:
            self.cleanup_image(image_path)

    def run(self) -> None:
        if not Config.validate():
            logging.error("Configuration validation failed")
            return
        logging.info(f"Starting Yellow Car Bot - runtime: {Config.MAX_RUNTIME_MINUTES} minutes")
        urls, current_index, current_stats = ShuffleStateManager.get_shuffled_urls()
        if not urls:
            logging.error("No webcam URLs available")
            return
        logging.info(f"Resuming from position {current_index}/{len(urls)} "
                     f"({current_index / len(urls) * 100:.1f}% of current cycle)")
        logging.info(f"All-time stats: {current_stats.get('total_processed', 0)} processed, "
                     f"{current_stats.get('total_posted', 0)} posted")
        processed_count = 0
        final_index = current_index
        try:
            for i in range(current_index, min(current_index + Config.IMAGES_PER_SESSION, len(urls))):
                if datetime.now() >= self.max_end_time:
                    logging.info("Time limit reached, stopping.")
                    break
                url = urls[i]
                try:
                    self.process_single_image(url, i)
                    processed_count += 1
                    final_index = i + 1
                    time.sleep(Config.REQUEST_DELAY)
                except RateLimitException:
                    break
                except Exception as e:
                    logging.error(f"Error processing image {i+1}: {e}")
                    continue
        except KeyboardInterrupt:
            logging.info("Interrupted by user, saving progress...")
        finally:
            stats_update = {
                "total_processed": self.session_stats['processed'],
                "total_posted": self.session_stats['posted']
            }
            ShuffleStateManager.update_progress(final_index, stats_update)
            self.print_summary(urls, final_index)

    def print_summary(self, urls: List[str], final_index: int) -> None:
        runtime = datetime.now() - self.start_time
        updated_stats = ShuffleStateManager.load_state()["stats"]
        print(f"\n{'=' * 25} SESSION SUMMARY {'=' * 25}")
        print(f"Runtime: {runtime.total_seconds():.1f}s ({runtime.total_seconds() / 60:.1f}m)")
        print(f"Images processed: {self.session_stats['processed']}")
        print(f"Yellow clusters found: {self.session_stats['yellow_clusters']}")
        print(f"AI confirmations: {self.session_stats['ai_confirmations']}")
        print(f"Cars posted to Bluesky: {self.session_stats['posted']}")
        print(f"Progress: {final_index}/{len(urls)} ({final_index / len(urls) * 100:.1f}%)")
        print(f"All-time totals: {updated_stats['total_processed']} processed, "
              f"{updated_stats['total_posted']} posted")
        if self.session_stats['processed'] > 0:
            yellow_rate = (self.session_stats['yellow_clusters'] / self.session_stats['processed']) * 100
            print(f"Yellow detection rate: {yellow_rate:.1f}%")
            if self.session_stats['yellow_clusters'] > 0:
                confirm_rate = (self.session_stats['ai_confirmations'] / self.session_stats['yellow_clusters']) * 100
                print(f"AI confirmation rate: {confirm_rate:.1f}%")

def main():
    try:
        bot = YellowCarBot()
        bot.run()
    except Exception as e:
        logging.critical(f"Critical error in main: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
