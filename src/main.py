#!/usr/bin/env python3
"""
Yellow Car Detection Bot (GitHub Models via Azure API version)
This version of the bot uses the new GitHub Models API endpoint to check for yellow cars.
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
    TOKEN = os.getenv("GITHUB_API_KEY") # This should be the new GitHub Models API key, not a regular PAT
    ENDPOINT = "https://models.github.ai"
    MODEL_NAME = "gpt-4o"
    TODAY_FOLDER = Path("today")
    WEBCAM_URLS_FILE = Path("valid_webcam_ids.txt")
    SHUFFLE_STATE_FILE = Path("shuffle_state.json")
    BSKY_HANDLE = os.getenv("BSKY_HANDLE")
    BSKY_PASSWORD = os.getenv("BSKY_PASSWORD")

    MAX_RUNTIME_MINUTES = 20
    IMAGES_PER_SESSION = 30
    YELLOW_THRESHOLD = 150
    MIN_CLUSTER_SIZE = 80
    API_TIMEOUT = 30
    REQUEST_DELAY = 1.0  # seconds

    @classmethod
    def validate(cls) -> bool:
        missing = []
        if not cls.TOKEN:
            missing.append("GITHUB_API_KEY")
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
    """Custom exception for rate limiting"""
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
                timeout=10, # Hardcoded timeout
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

def get_image_data_url(image_file, image_format):
    try:
        with open(image_file, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        return f"data:image/{image_format};base64,{image_base64}"
    except Exception as e:
        logging.error(f"Could not read '{image_file}': {e}")
        return None

class AIDetector:
    @staticmethod
    def ask_ai_if_yellow_car(image_path: Path) -> Optional[str]:
        if not Config.TOKEN:
            logging.error("GitHub Models API token is not defined")
            return None

        image_data_url = get_image_data_url(image_path, "jpg")
        if not image_data_url:
            return None

        headers = {
            "Authorization": f"Bearer {Config.TOKEN}",
            "Content-Type": "application/json"
        }
        body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in identifying vehicles. Answer with only 'yes' or 'no'."},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": "Is there a yellow VEHICLE (car, truck, van, bus, or motorcycle) visible in this traffic camera image? Look specifically for yellow-colored vehicles with wheels, windows, and automotive features. DO NOT count yellow road markings, yellow lines on pavement, yellow traffic signs, yellow construction equipment that is stationary, or any other non-vehicle yellow objects. Only respond 'yes' if you can clearly identify a yellow motor vehicle. Answer only 'yes' or 'no'."},
                    {"type": "image_url", "image_url": {"url": image_data_url, "detail": "low"}}
                ]}
            ],
            "model": Config.MODEL_NAME,
            "max_tokens": 10
        }

        try:
            resp = requests.post(f"{Config.ENDPOINT}/chat/completions", json=body, headers=headers, timeout=Config.API_TIMEOUT)

            if resp.status_code == 429:
                logging.warning("🚫 Rate limit hit (429):")
                logging.warning("Stopping session to preserve API minutes")
                raise RateLimitException("Rate limit reached")

            if resp.status_code != 200:
                logging.error(f"GitHub Models API error: {resp.status_code} - {resp.text}")
                return None

            data = resp.json()
            return data["choices"][0]["message"]["content"].strip().lower()

        except RateLimitException:
            raise
        except Exception as e:
            logging.error(f"Error calling AI endpoint: {e}")
            return None

class BlueskyPoster:
    @staticmethod
    def post_image(image_path: Path, alt_text: str) -> bool:
        if not Config.BSKY_HANDLE or not Config.BSKY_PASSWORD:
            logging.error("Bluesky credentials not defined")
            return False

        try:
            client = Client()
            client.login(Config.BSKY_HANDLE.strip(), Config.BSKY_PASSWORD.strip())

            image_data_url = get_image_data_url(image_path, "jpg")
            if not image_data_url:
                return False

            header, encoded = image_data_url.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            blob = client.upload_blob(image_bytes).blob

            client.app.bsky.feed.post.create(
                repo=client.me.did,
                record=models.AppBskyFeedPost.Record(
                    text="GUL BIL! 🟡🚗",
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
                logging.info(f"🟡 Yellow cluster detected! Checking with AI...")
                
                try:
                    ai_response = AIDetector.ask_ai_if_yellow_car(image_path)
                    logging.info(f"AI response: {ai_response}")
                    
                    if ai_response and "yes" in ai_response:
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
            final_index = current_index + processed_count
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
