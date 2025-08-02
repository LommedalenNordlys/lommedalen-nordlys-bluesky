"""
Yellow Car Detection Bot
Monitors webcam feeds for yellow vehicles and posts findings to Bluesky
Uses Hugging Face Florence-2-large model for vehicle detection
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
    # API Configuration
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    HF_MODEL_ENDPOINT = "https://api-inference.huggingface.co/models/microsoft/Florence-2-large"
    
    # Bluesky Configuration
    BSKY_HANDLE = os.getenv("BSKY_HANDLE")
    BSKY_PASSWORD = os.getenv("BSKY_PASSWORD")
    
    # File paths
    TODAY_FOLDER = Path("today")
    WEBCAM_URLS_FILE = Path("valid_webcam_ids.txt")
    SHUFFLE_STATE_FILE = Path("shuffle_state.json")
    
    # Runtime limits
    MAX_RUNTIME_MINUTES = 20
    IMAGES_PER_SESSION = 30
    
    # Image processing
    YELLOW_THRESHOLD = 150
    MIN_CLUSTER_SIZE = 80
    DOWNLOAD_TIMEOUT = 10
    API_TIMEOUT = 30
    
    # Rate limiting
    REQUEST_DELAY = 1.0  # seconds between requests
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
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
    """Custom exception for rate limiting"""
    pass

class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass

class ShuffleStateManager:
    """Manages the shuffle state for webcam URLs"""
    
    @staticmethod
    def load_state() -> Dict:
        """Load shuffle state from file"""
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
                
            # Validate state structure
            if not isinstance(state.get("shuffled_urls"), list):
                logging.warning("Invalid shuffle state, using defaults")
                return default_state
                
            # Merge with defaults to ensure all keys exist
            for key, value in default_state.items():
                if key not in state:
                    state[key] = value
                    
            return state
            
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Could not load shuffle state: {e}")
            return default_state
    
    @staticmethod
    def save_state(state: Dict) -> None:
        """Save shuffle state to file"""
        try:
            with open(Config.SHUFFLE_STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except IOError as e:
            logging.error(f"Could not save shuffle state: {e}")
    
    @staticmethod
    def get_shuffled_urls() -> Tuple[List[str], int, Dict]:
        """Get shuffled URLs and current processing state"""
        try:
            with open(Config.WEBCAM_URLS_FILE, "r", encoding='utf-8') as f:
                all_urls = [line.strip() for line in f if line.strip()]
        except IOError as e:
            logging.error(f"Could not read webcam URLs file: {e}")
            return [], 0, {}

        state = ShuffleStateManager.load_state()

        # Check if we need to reshuffle
        if (not state["shuffled_urls"] or 
            state["current_index"] >= len(state["shuffled_urls"]) or
            len(state["shuffled_urls"]) != len(all_urls)):
            
            logging.info(f"Shuffling {len(all_urls)} webcam URLs for fair processing")
            state["shuffled_urls"] = all_urls.copy()
            random.shuffle(state["shuffled_urls"])
            state["current_index"] = 0
            state["cycle_count"] = state.get("cycle_count", 0) + 1
            logging.info(f"Starting cycle #{state['cycle_count']}")
            ShuffleStateManager.save_state(state)

        return state["shuffled_urls"], state["current_index"], state["stats"]
    
    @staticmethod
    def update_progress(new_index: int, stats_update: Optional[Dict] = None) -> None:
        """Update processing progress and stats"""
        state = ShuffleStateManager.load_state()
        state["current_index"] = new_index
        
        if stats_update:
            for key, value in stats_update.items():
                state["stats"][key] = state["stats"].get(key, 0) + value
                
        ShuffleStateManager.save_state(state)

class ImageProcessor:
    """Handles image downloading and yellow color detection"""
    
    @staticmethod
    def download_image(url: str, dest: Path) -> bool:
        """Download image from URL with error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
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
                logging.debug(f"Failed to download {url}: Status {resp.status_code}")
                return False
                
        except requests.RequestException as e:
            logging.debug(f"Network error downloading {url}: {e}")
            return False
        except IOError as e:
            logging.debug(f"File error saving {dest}: {e}")
            return False
    
    @staticmethod
    def find_yellow_clusters(image_path: Path) -> bool:
        """Detect yellow color clusters in image using optimized sampling"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get image dimensions
                width, height = img.size
                
                # Use more efficient sampling for large images
                step_size = max(1, min(width, height) // 200)  # Adaptive step size
                yellow_count = 0
                total_samples = 0
                
                # Sample pixels in a grid pattern
                for y in range(0, height, step_size):
                    for x in range(0, width, step_size):
                        try:
                            r, g, b = img.getpixel((x, y))
                            total_samples += 1
                            
                            # Check for yellow-ish color (high R and G, low B)
                            if (r > Config.YELLOW_THRESHOLD and 
                                g > Config.YELLOW_THRESHOLD and 
                                b < 100):
                                yellow_count += 1
                                
                        except IndexError:
                            continue  # Skip invalid coordinates
                
                # Calculate percentage of yellow pixels
                if total_samples == 0:
                    return False
                    
                yellow_percentage = (yellow_count / total_samples) * 100
                return yellow_count >= Config.MIN_CLUSTER_SIZE or yellow_percentage > 5.0
                
        except (IOError, OSError) as e:
            logging.debug(f"Error processing image {image_path}: {e}")
            return False

class AIDetector:
    """Handles AI-based vehicle detection using Hugging Face API"""
    
    @staticmethod
    def get_image_data_url(image_path: Path) -> Optional[str]:
        """Convert image to base64 data URL"""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Validate image size (HF has limits)
            if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
                logging.warning(f"Image {image_path} too large for API")
                return None
                
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
            
        except (IOError, OSError) as e:
            logging.error(f"Could not read image '{image_path}': {e}")
            return None
    
    @staticmethod
    def detect_yellow_vehicle(image_path: Path) -> Optional[str]:
        """Use Hugging Face Florence-2 model to detect yellow vehicles"""
        image_data_url = AIDetector.get_image_data_url(image_path)
        if not image_data_url:
            return None

        headers = {
            "Authorization": f"Bearer {Config.HF_API_TOKEN}",
            "Content-Type": "application/json"
        }

        # Simplified payload for Florence-2
        payload = {
            "inputs": {
                "image": image_data_url,
                "text": "<OD>"  # Object Detection task
            },
            "parameters": {
                "max_new_tokens": 100
            }
        }

        try:
            response = requests.post(
                Config.HF_MODEL_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=Config.API_TIMEOUT
            )

            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After', '60')
                logging.warning(f"Rate limit hit. Retry after {retry_after} seconds")
                raise RateLimitException(f"API rate limit exceeded. Retry after {retry_after}s")

            if response.status_code == 503:
                logging.warning("Model is loading, this may take a few minutes")
                return None

            if response.status_code != 200:
                logging.error(f"HF API error: {response.status_code} - {response.text}")
                return None

            result = response.json()
            
            # Parse Florence-2 object detection response
            if isinstance(result, list) and len(result) > 0:
                detection_text = result[0].get('generated_text', '').lower()
                
                # Look for vehicle-related keywords and yellow color
                vehicle_keywords = ['car', 'truck', 'van', 'bus', 'vehicle', 'taxi']
                yellow_keywords = ['yellow', 'gold', 'amber']
                
                has_vehicle = any(keyword in detection_text for keyword in vehicle_keywords)
                has_yellow = any(keyword in detection_text for keyword in yellow_keywords)
                
                if has_vehicle and has_yellow:
                    return "yes"
                elif has_vehicle:
                    return "vehicle_found_no_yellow"
                else:
                    return "no"
            
            logging.debug(f"Unexpected API response format: {result}")
            return None

        except RateLimitException:
            raise
        except requests.RequestException as e:
            logging.error(f"Network error calling HF API: {e}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Error parsing HF API response: {e}")
            return None

class BlueskyPoster:
    """Handles posting to Bluesky social network"""
    
    @staticmethod
    def post_image(image_path: Path, alt_text: str) -> bool:
        """Post image to Bluesky with error handling"""
        try:
            client = Client()
            client.login(Config.BSKY_HANDLE.strip(), Config.BSKY_PASSWORD.strip())

            # Read and upload image
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            blob = client.upload_blob(image_data).blob

            # Create post
            post_text = "GUL BIL! 🟡🚗"  # Yellow car in Norwegian with emojis
            
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
    """Main bot class orchestrating the yellow car detection workflow"""
    
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
        """Safely remove temporary image file"""
        try:
            if image_path.exists():
                image_path.unlink()
        except OSError as e:
            logging.debug(f"Could not remove {image_path}: {e}")
    
    def process_single_image(self, url: str, index: int) -> bool:
        """Process a single webcam image"""
        timestamp = int(time.time())
        image_name = f"cam_{index + 1}_{timestamp}.jpg"
        image_path = Config.TODAY_FOLDER / image_name
        
        try:
            # Download image
            logging.info(f"Processing {index + 1}: downloading image...")
            if not ImageProcessor.download_image(url, image_path):
                return False
            
            self.session_stats['processed'] += 1
            
            # Check for yellow clusters
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
                    
                except RateLimitException as e:
                    logging.warning(f"Rate limit reached: {e}")
                    raise  # Re-raise to stop processing
            
            return False
            
        finally:
            # Always cleanup the temporary image
            self.cleanup_image(image_path)
    
    def run(self) -> None:
        """Main bot execution loop"""
        if not Config.validate():
            logging.error("Configuration validation failed")
            return
        
        logging.info(f"Starting Yellow Car Bot - max runtime: {Config.MAX_RUNTIME_MINUTES} minutes")
        
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
                # Check time limit
                if datetime.now() >= self.max_end_time:
                    logging.info("Time limit reached, stopping gracefully")
                    break
                
                url = urls[i]
                
                try:
                    self.process_single_image(url, i)
                    processed_count += 1
                    final_index = i + 1
                    
                    # Rate limiting delay
                    time.sleep(Config.REQUEST_DELAY)
                    
                except RateLimitException:
                    logging.warning("Stopping due to rate limit to preserve resources")
                    break
                except Exception as e:
                    logging.error(f"Error processing image {i+1}: {e}")
                    continue
        
        except KeyboardInterrupt:
            logging.info("Interrupted by user, saving progress...")
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
        finally:
            # Save progress
            stats_update = {
                "total_processed": self.session_stats['processed'],
                "total_posted": self.session_stats['posted']
            }
            ShuffleStateManager.update_progress(final_index, stats_update)
            self.print_summary(urls, final_index)
    
    def print_summary(self, urls: List[str], final_index: int) -> None:
        """Print session summary statistics"""
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
        
        # Additional metrics
        if self.session_stats['processed'] > 0:
            yellow_rate = (self.session_stats['yellow_clusters'] / self.session_stats['processed']) * 100
            print(f"Yellow detection rate: {yellow_rate:.1f}%")
            
            if self.session_stats['yellow_clusters'] > 0:
                confirm_rate = (self.session_stats['ai_confirmations'] / self.session_stats['yellow_clusters']) * 100
                print(f"AI confirmation rate: {confirm_rate:.1f}%")

def main():
    """Entry point"""
    try:
        bot = YellowCarBot()
        bot.run()
    except Exception as e:
        logging.critical(f"Critical error in main: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    exit(main())