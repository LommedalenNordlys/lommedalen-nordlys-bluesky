
import os
import sys
import requests
import base64
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from atproto import Client, models

load_dotenv()


TOKEN = os.getenv("KEY_GITHUB_TOKEN")
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o"
TODAY_FOLDER = Path("today")
TODAY_FOLDER.mkdir(exist_ok=True)
WEBCAM_URLS_FILE = Path("valid_webcam_ids.txt")
BSKY_HANDLE = os.getenv("BSKY_HANDLE")
BSKY_PASSWORD = os.getenv("BSKY_PASSWORD")

logging.basicConfig(level=logging.INFO)

def download_image(url, dest):
    try:
        resp = requests.get(url, allow_redirects=True, timeout=15)
        if resp.status_code == 200:
            with open(dest, "wb") as f:
                f.write(resp.content)
            return True
        else:
            logging.warning(f"Failed to download {url}: Status {resp.status_code}")
            return False
    except Exception as e:
        logging.warning(f"Exception downloading {url}: {e}")
        return False

def find_yellow_clusters(image_path, min_cluster_size=100):
    try:
        img = Image.open(image_path).convert('RGB')
        pixels = img.load()
        width, height = img.size
        yellow_pixels = []
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                # Simple yellow detection: high R and G, low B
                if r > 180 and g > 180 and b < 120:
                    yellow_pixels.append((x, y))
        # Cluster detection: if enough yellow pixels
        return len(yellow_pixels) >= min_cluster_size
    except Exception as e:
        logging.warning(f"Error processing {image_path}: {e}")
        return False

def get_image_data_url(image_file, image_format):
    try:
        with open(image_file, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        return f"data:image/{image_format};base64,{image_base64}"
    except Exception as e:
        logging.error(f"Could not read '{image_file}': {e}")
        return None

def ask_ai_if_yellow_car(image_path):
    import time
    if not TOKEN:
        logging.error("Azure API token is not defined")
        return None
    image_data_url = get_image_data_url(image_path, "jpg")
    if not image_data_url:
        return None
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    body = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions about images."},
            {"role": "user", "content": [
                {"type": "text", "text": "Is there a yellow car in this image? Answer yes or no."},
                {"type": "image_url", "image_url": {"url": image_data_url, "detail": "low"}}
            ]}
        ],
        "model": MODEL_NAME
    }
    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = requests.post(f"{ENDPOINT}/chat/completions", json=body, headers=headers, timeout=30)
            if resp.status_code == 429 or (resp.status_code != 200 and "RateLimitReached" in resp.text):
                # Try to extract wait time from response
                try:
                    data = resp.json()
                    msg = data.get("error", {}).get("message", "")
                    import re
                    wait_match = re.search(r'wait (\d+) seconds', msg)
                    wait_seconds = int(wait_match.group(1)) if wait_match else 20
                    # Cap wait_seconds to a reasonable maximum (e.g., 120 seconds)
                    if wait_seconds > 120:
                        logging.warning(f"Wait time from API too high ({wait_seconds}s), capping to 120s.")
                        wait_seconds = 120
                except Exception:
                    wait_seconds = 20
                logging.warning(f"Rate limit hit, waiting {wait_seconds} seconds before retrying...")
                time.sleep(wait_seconds)
                continue
            if resp.status_code != 200:
                logging.error(f"Azure API error response: {resp.text}")
                return None
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip().lower()
        except Exception as e:
            logging.error(f"Error calling AI endpoint: {e}")
            return None
    logging.error("Max retries reached for AI endpoint due to rate limiting.")
    return None


def post_to_bluesky(image_path, alt_text):
    if not BSKY_HANDLE or not BSKY_PASSWORD:
        logging.error("Bluesky handle or password is not defined")
        return False
    try:
        client = Client()
        client.login(BSKY_HANDLE.strip(), BSKY_PASSWORD.strip())
        image_data_url = get_image_data_url(image_path, "jpg")
        if not image_data_url:
            logging.error(f"Could not get data URL for {image_path}")
            return False
        header, encoded = image_data_url.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        blob = client.upload_blob(image_bytes).blob
        client.app.bsky.feed.post.create(
            repo=client.me.did,
            record=models.AppBskyFeedPost.Record(
                text="GUL BIL!",
                created_at=datetime.utcnow().isoformat() + "Z",
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
        logging.info(f"Posted {image_path} to Bluesky with text 'GUL BIL!'")
        return True
    except Exception as e:
        logging.error(f"Error posting to Bluesky: {e}")
        return False

def main():
    if not WEBCAM_URLS_FILE.exists():
        logging.error(f"Webcam URLs file not found: {WEBCAM_URLS_FILE}")
        return
    with open(WEBCAM_URLS_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    results = []
    for idx, url in enumerate(urls):
        image_name = f"webcam_{idx+1}.jpg"
        image_path = TODAY_FOLDER / image_name
        logging.info(f"Downloading {url} -> {image_path}")
        if not download_image(url, image_path):
            continue
        if find_yellow_clusters(image_path):
            logging.info(f"Yellow cluster detected in {image_name}, sending to AI...")
            ai_response = ask_ai_if_yellow_car(image_path)
            results.append((url, image_name, ai_response))
            logging.info(f"AI response for {image_name}: {ai_response}")
            if ai_response and "no" not in ai_response and ("yes" in ai_response or "yellow car" in ai_response):
                post_to_bluesky(image_path, alt_text="GUL BIL!")
        else:
            logging.info(f"No yellow cluster detected in {image_name}")
    # Print summary
    print("\nSummary of images with yellow clusters and AI responses:")
    for url, image_name, ai_response in results:
        print(f"{image_name}: {url}\nAI: {ai_response}\n")

if __name__ == "__main__":
    main()
