# Lommedalen Aurora Borealis Bot 🌌

An automated bot that monitors webcams in the Lommedalen area for aurora borealis (northern lights) and posts them to Bluesky when detected. It performs a darkness check (astronomical → nautical → civil twilight) and a Kp index threshold check before any image processing to avoid wasting resources.

## How It Works

1. **Darkness Gate** – Fetches sunrise/sunset & twilight boundaries; exits early if it isn't dark enough.
2. **Kp Index Gate** – Checks current planetary Kp index from NOAA; exits if below threshold (default: 3).
3. **Image Download** – Pulls current frames from configured public webcams.
4. **Color Pre-filter** – Fast heuristic for aurora-like greens/purples to reduce AI calls.
5. **AI Confirmation** – GPT‑4o Vision answers yes/no for aurora presence; handles resize & rate limits.
6. **Posting** – Uploads confirmed aurora with location metadata to Bluesky.

## Monitoring Locations (Current)

Defined per view in `webcam_locations.json`:
- Tryvann North West
- Tryvann North
- Tryvann South East
- Tryvann South
- Tryvann West (shares camera with South for now)
- Sylling North
- Sylling South

## Features

- **Darkness Gate**: Skips processing entirely when twilight data indicates it's still light.
- **Kp Index Gate**: Requires minimum planetary Kp index (default: 3) from NOAA to proceed.
- **Location Tracking**: Each post includes the view where aurora was detected.
- **Fair Processing**: Shuffles webcam order to vary observation times.
- **Color Pre-filtering**: Quick scan for aurora-like colors before AI check.
- **Rate Limit Handling**: Gracefully handles API 429 & image size 413 responses.
- **Persistent State**: Remembers progress between runs.
- **Statistics Tracking**: Detection and posting counters.

## Setup

### 1. Repository Structure
```
your-repo/
├── src/
│   └── main.py
├── webcam_locations.json
├── requirements.txt
├── .github/
│   └── workflows/
│       └── aurora-bot.yml
└── README.md
```

### 2. Required Files

**requirements.txt**:
```txt
requests
python-dotenv
Pillow
atproto
```

**webcam_locations.json**:
Contains webcam URLs organized by location (see included file).

### 3. GitHub Secrets

Set up these secrets in your repository (Settings → Secrets and variables → Actions):

- `BSKY_HANDLE`: Your Bluesky handle (e.g., `username.bsky.social`)
- `BSKY_PASSWORD`: Your Bluesky app password (not your main password!)
- `KEY_GITHUB_TOKEN`: Azure AI token for GPT-4o vision API

### 4. Bluesky App Password

1. Go to Settings → Privacy and Security → App Passwords
2. Create a new app password for this bot
3. Use this password (not your main password) in the `BSKY_PASSWORD` secret

## Configuration

Edit these variables in `src/main.py`:

```python
MAX_RUNTIME_MINUTES = 20    # Max runtime per session
IMAGES_PER_SESSION = 30     # Images to process per session
MIN_KP_INDEX = 3            # Minimum Kp index required (0-9 scale)
## Schedule

GitHub Actions currently runs every 30 minutes during typical dark hours. Because the code now performs a dynamic darkness check, you can optionally widen the cron schedule (e.g. run 24/7) and rely on fast early exits.
## How the Detection Works

## Pipeline Overview

1. Darkness check (sunrise-sunset API for twilight boundaries)
2. Kp index check (NOAA real-time solar wind data; minimum 3.0)
3. Download webcam images
4. Color heuristic pre-filter (green/purple detection)
5. AI verification (GPT-4o Vision yes/no) with automatic resize & retry
6. Post confirmed aurora to Bluesky with location

## Aurora Detection Details

The bot looks for:
- **Green aurora**: Bright green dominant colors in the sky
- **Purple/Pink aurora**: Reddish-purple glows
- **Night sky context**: Only valid in dark conditions
- **Natural patterns**: Distinguishes from artificial lights
## Troubleshooting

### Common Issues

**"Not dark enough"** – Expected during daylight/twilight; bot exits quickly.
**"Kp below threshold"** – Planetary Kp index is too low for aurora activity; bot exits to save resources.
**"Kp index unavailable"** – NOAA API temporarily down; bot proceeds anyway (fail-open).
**"AI response: no"** – Color heuristic can flag vegetation or lights; AI filter reduces false positives.
**"Rate limit (429)"** – Azure quota exceeded; bot skips more AI calls until next run.
**"413 payload too large"** – Image auto-compressed & retried.
**"Webcam unavailable"** – Temporary outage; skipped.
**"No aurora colors detected"** – Normal; aurora is rare.

### Best Conditions for Aurora

- **Dark skies**: Late evening through early morning
- **High KP-index**: Check aurora forecasts
- **Clear weather**: Clouds will block view
- **Winter months**: Longer dark periods

### Manual Testing

Trigger a manual run:
1. Go to Actions tab
2. Select "Aurora Borealis Bot"
3. Click "Run workflow"

## File Descriptions

- **`src/main.py`**: Main bot logic
- **`webcam_locations.json`**: Webcam URLs with location data
- **`requirements.txt`**: Python dependencies
- **`.github/workflows/`**: GitHub Actions workflow
- **`shuffle_state.json`**: Persistent state (auto-generated)

## Webcam Sources

- Tryvann: yr.no public webcams (directional views)
- Sylling: yr.no public webcams
## Resources

- [NOAA OVATION Aurora Forecast](https://services.swpc.noaa.gov/json/ovation_aurora_latest.json) - Real-time data used by the bot
- [Space Weather Prediction Center](https://www.swpc.noaa.gov/) - Official NOAA space weather site
- [Aurora forecast for Norway](https://www.aurora-service.eu/aurora-forecast/) - Additional forecast source
- [Bluesky API Documentation](https://docs.bsky.app/) - Bluesky API reference

- Only processes public webcam feeds
- No personal data collected or stored
- Images deleted immediately after processing
- Respects API rate limits and terms of service

## Contributing

Feel free to:
- Add/remove webcam views in `webcam_locations.json`
- Tune color thresholds in `check_for_aurora_colors()` function
- Adjust MIN_KP threshold (set via env var or default in code)
- Add posting cooldowns or rate limiting
- Tune darkness sensitivity (astronomical vs nautical twilight)

## Resources

- [Space Weather Prediction Center](https://www.swpc.noaa.gov/)
- [Aurora forecast for Norway](https://www.aurora-service.eu/aurora-forecast/)
- [Bluesky API Documentation](https://docs.bsky.app/)

## License

This project is for educational and aurora spotting purposes. Please respect:
- Webcam usage terms
- Bluesky community guidelines
- API rate limits and quotas

---

**Lykke til med nordlys-jakten!** 🌌✨
