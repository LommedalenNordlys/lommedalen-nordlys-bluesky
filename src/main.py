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
        """Detect yellow color clusters in image, focusing on vehicle-sized areas"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get image dimensions
                width, height = img.size
                
                # Create a grid to analyze regions (looking for vehicle-sized yellow areas)
                grid_size = 20  # Analyze in 20x20 pixel blocks
                yellow_regions = []
                
                for y in range(0, height - grid_size, grid_size):
                    for x in range(0, width - grid_size, grid_size):
                        yellow_count = 0
                        total_pixels = 0
                        
                        # Sample pixels in this grid region
                        for dy in range(0, grid_size, 2):
                            for dx in range(0, grid_size, 2):
                                try:
                                    px_x, px_y = x + dx, y + dy
                                    if px_x < width and px_y < height:
                                        r, g, b = img.getpixel((px_x, px_y))
                                        total_pixels += 1
                                        
                                        # More refined yellow detection
                                        # High red and green, low blue, but also check ratios
                                        if (r > Config.YELLOW_THRESHOLD and 
                                            g > Config.YELLOW_THRESHOLD and 
                                            b < 120 and
                                            r + g > 2 * b):  # Ensure yellow-ish ratio
                                            yellow_count += 1
                                            
                                except IndexError:
                                    continue
                        
                        if total_pixels > 0:
                            yellow_percentage = (yellow_count / total_pixels) * 100
                            
                            # If this region has significant yellow content
                            if yellow_percentage > 25 and yellow_count > 10:
                                yellow_regions.append({
                                    'x': x, 'y': y, 'size': grid_size,
                                    'yellow_pixels': yellow_count,
                                    'percentage': yellow_percentage
                                })
                
                # Check if we have enough significant yellow regions
                # Could be one large vehicle or multiple smaller elements
                if not yellow_regions:
                    return False
                
                total_yellow_pixels = sum(region['yellow_pixels'] for region in yellow_regions)
                
                # Log the regions found for debugging
                if len(yellow_regions) > 0:
                    logging.debug(f"Found {len(yellow_regions)} yellow regions with total {total_yellow_pixels} yellow pixels")
                
                # Criteria for detection:
                # 1. At least one substantial region, OR
                # 2. Multiple smaller regions that could form a vehicle
                substantial_regions = [r for r in yellow_regions if r['yellow_pixels'] > 20]
                
                return (len(substantial_regions) >= 1 or 
                        (len(yellow_regions) >= 2 and total_yellow_pixels > Config.MIN_CLUSTER_SIZE))
                
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
        """Attempt multiple AI models to detect yellow vehicles"""
        
        # Try multiple models in order of preference
        models_to_try = [
            "nlpconnect/vit-gpt2-image-captioning",
            "Salesforce/blip-image-captioning-base", 
            "microsoft/git-base-coco",
        ]
        
        for model_name in models_to_try:
            try:
                result = AIDetector._try_model(image_path, model_name)
                if result is not None:
                    return result
                logging.debug(f"Model {model_name} returned None, trying next...")
            except RateLimitException:
                raise  # Don't try other models if rate limited
            except Exception as e:
                logging.debug(f"Model {model_name} failed: {e}, trying next...")
                continue
        
        # If all models fail, fall back to enhanced color analysis
        logging.warning("All AI models failed, using enhanced color analysis fallback")
        return AIDetector._enhanced_color_analysis(image_path)
    
    @staticmethod
    def _try_model(image_path: Path, model_name: str) -> Optional[str]:
        """Try a specific model for image captioning"""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            if len(image_data) > 10 * 1024 * 1024:
                return None
                
        except (IOError, OSError) as e:
            logging.error(f"Could not read image '{image_path}': {e}")
            return None

        endpoint = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {Config.HF_API_TOKEN}"}

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                data=image_data,
                timeout=Config.API_TIMEOUT
            )

            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After', '60')
                logging.warning(f"Rate limit hit on {model_name}")
                raise RateLimitException(f"API rate limit exceeded. Retry after {retry_after}s")

            if response.status_code == 503:
                logging.debug(f"Model {model_name} is loading")
                return None

            if response.status_code == 404:
                logging.debug(f"Model {model_name} not available via Inference API")
                return None

            if response.status_code != 200:
                logging.debug(f"Model {model_name} error: {response.status_code}")
                return None

            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                caption = result[0].get('generated_text', '').lower().strip()
                if caption:
                    logging.info(f"[{model_name}] Caption: '{caption}'")
                    return AIDetector._analyze_caption(caption)
            
            return None

        except RateLimitException:
            raise
        except Exception as e:
            logging.debug(f"Error with model {model_name}: {e}")
            return None
    
    @staticmethod
    def _analyze_caption(caption: str) -> str:
        """Analyze caption for yellow vehicle indicators"""
        vehicle_keywords = [
            'car', 'truck', 'van', 'bus', 'vehicle', 'taxi', 'automobile', 
            'suv', 'sedan', 'coupe', 'hatchback', 'convertible'
        ]
        yellow_keywords = ['yellow', 'gold', 'golden', 'amber', 'bright yellow', 'lemon']
        
        # Check for explicit yellow vehicle phrases first
        yellow_vehicle_phrases = [
            'yellow car', 'yellow truck', 'yellow van', 'yellow bus', 
            'yellow taxi', 'taxi cab', 'yellow vehicle', 'gold car',
            'golden car', 'bright yellow car', 'yellow sedan', 'yellow suv'
        ]
        
        has_yellow_vehicle_phrase = any(phrase in caption for phrase in yellow_vehicle_phrases)
        
        if has_yellow_vehicle_phrase:
            logging.info(f"✅ Explicit yellow vehicle phrase found")
            return "yes"
        
        # Check for separate mentions
        has_vehicle = any(keyword in caption for keyword in vehicle_keywords)
        has_yellow = any(keyword in caption for keyword in yellow_keywords)
        
        if has_vehicle and has_yellow:
            # Simple context check
            if len(caption.split()) <= 12:  # Short caption, likely related
                logging.info(f"✅ Vehicle and yellow in short caption")
                return "yes"
            else:
                logging.info(f"🟡 Vehicle and yellow found but context unclear")
                return "maybe"
        elif has_vehicle:
            logging.debug(f"❌ Vehicle found but no yellow")
            return "vehicle_found_no_yellow"
        else:
            logging.debug(f"❌ No vehicle detected")
            return "no"
    
    @staticmethod 
    def _enhanced_color_analysis(image_path: Path) -> str:
        """Enhanced fallback color analysis when AI fails"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                width, height = img.size
                center_region = {
                    'x1': width // 4,
                    'y1': height // 4, 
                    'x2': 3 * width // 4,
                    'y2': 3 * height // 4
                }
                
                yellow_pixels = 0
                total_pixels = 0
                
                # Focus on center region where vehicles are more likely
                for y in range(center_region['y1'], center_region['y2'], 3):
                    for x in range(center_region['x1'], center_region['x2'], 3):
                        try:
                            r, g, b = img.getpixel((x, y))
                            total_pixels += 1
                            
                            # More refined yellow detection
                            if (r > 180 and g > 180 and b < 100 and 
                                abs(r - g) < 50 and r + g > 2.5 * b):
                                yellow_pixels += 1
                                
                        except IndexError:
                            continue
                
                if total_pixels == 0:
                    return "no"
                
                yellow_percentage = (yellow_pixels / total_pixels) * 100
                
                # Conservative thresholds for fallback
                if yellow_percentage > 15 and yellow_pixels > 30:
                    logging.info(f"🟡 Fallback: Strong yellow signal ({yellow_percentage:.1f}%)")
                    return "maybe"  # Conservative - don't post without AI confirmation
                else:
                    return "no"
                    
        except Exception as e:
            logging.error(f"Fallback color analysis failed: {e}")
            return "no"

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