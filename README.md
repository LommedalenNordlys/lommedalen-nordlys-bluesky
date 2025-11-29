# Aurora Borealis Detection Bot 🌌

An automated bot that monitors webcams for aurora borealis (northern lights) and posts them to Bluesky when detected. Designed to be easily forked and configured for any location with public webcams and YR.no coverage.

## 🚀 Quick Start for Your Location

### 1. Fork this Repository

Click the "Fork" button on GitHub to create your own copy.

### 2. Configure for Your Location

Copy the example configuration files:
```bash
cp config.example.json config.json
cp webcam_locations.example.json webcam_locations.json
```

Edit `config.json` with your location details:

```json
{
  "location": {
    "name": "Your Location Name",
    "latitude": 59.95,
    "longitude": 10.466667,
    "yr_location_id": "1-72662"
  },
  "aurora": {
    "min_kp_index": 3,
    "max_runtime_minutes": 20,
    "images_per_session": 30
  }
}
```

**Finding Your YR.no Location ID:**
1. Go to [YR.no](https://www.yr.no/)
2. Search for your location
3. Look at the URL: `https://www.yr.no/en/forecast/daily-table/1-72662/Norway/Oslo/Oslo/Oslo`
4. Your location ID is the number after `/daily-table/` (e.g., `1-72662`)

**Setting Your Coordinates:**
- Use [latlong.net](https://www.latlong.net/) to find precise coordinates
- Coordinates should match or be close to your YR.no location for accurate aurora forecasts

### 3. Configure Webcams

Edit `webcam_locations.json` with your public webcam URLs:

```json
{
    "webcams": [
        {
            "place": "Mountain North View",
            "urls": ["https://example.com/webcam1.jpg"]
        },
        {
            "place": "Valley South View",
            "urls": ["https://example.com/webcam2.jpg"]
        }
    ]
}
```

**Finding Public Webcams:**
- Check your local weather service websites  
- Search for "[your location] live webcam"
- Look for tourism boards or ski resort webcams
- Ensure URLs end in `.jpg` or `.jpeg` for direct image access

### 4. Set Up GitHub Secrets

Go to your repository → Settings → Secrets and variables → Actions

Add these secrets:

- `BSKY_HANDLE`: Your Bluesky handle (e.g., `username.bsky.social`)
- `BSKY_PASSWORD`: Your Bluesky app password (Settings → App Passwords)
- `KEY_GITHUB_TOKEN`: Azure AI token for GPT-4o vision API
- `PAT_TOKEN`: GitHub Personal Access Token (for automated commits)

### 5. Enable GitHub Actions

1. Go to Actions tab in your repository
2. Enable workflows
3. The bot will run automatically during dark hours

### 6. Test Your Setup

Manually trigger a test run:
1. Go to Actions tab
2. Select "Lommedalen Aurora Borealis Bot"
3. Click "Run workflow"

## 📋 Configuration Reference

### Kp Index Settings

- `min_kp_index: 2` - Very sensitive, many false positives, best for high latitudes (>65°N)
- `min_kp_index: 3` - **Recommended default**, good balance
- `min_kp_index: 4` - Conservative, fewer detections but higher quality
- `min_kp_index: 5+` - Only major aurora events

### Location-Specific Tips

**Norway/Northern Scandinavia**
- YR.no coverage is excellent
- Use `min_kp_index: 3` (default)
- Winter months (Oct-Mar) have best conditions

**Scotland/Northern UK**
- Check YR.no availability first
- Increase `min_kp_index: 4` (less frequent events)
- Best months: Sep-Mar

**Canada/Alaska**
- YR.no may not cover your area
- Verify location ID availability before setup

## 🔧 How It Works

### Detection Pipeline

1. **Darkness Gate** – Checks astronomical twilight; exits if not dark
2. **Kp Index Gate** – Queries YR.no aurora forecast; exits if below threshold
3. **Image Download** – Fetches frames from configured webcams
4. **Color Pre-filter** – Fast heuristic for aurora colors (green/purple)
5. **AI Confirmation** – GPT-4o Vision validates aurora presence
6. **Posting** – Uploads to Bluesky with location metadata

### Resource Management

- Bash scripts for speed (~10x faster than Python for checks)
- Early exits save processing time when conditions unfavorable
- Color pre-filter reduces AI API calls by 70-80%
- Automatic schedule adjustment based on sunset/sunrise times

## 🛠️ Advanced Configuration

### Environment Variables Override

You can override config.json values with environment variables:

```bash
MIN_KP=4
MAX_RUNTIME_MINUTES=30
IMAGES_PER_SESSION=50
CONFIG_FILE=config.custom.json
```

### Custom AI Model

Edit `config.json` to use different models:

```json
{
  "api": {
    "azure_endpoint": "https://your-endpoint.com",
    "model_name": "gpt-4o-mini"
  }
}
```

## 📊 Monitoring

### KP Index Logs
Location: `kp_data/YYYY/MM/YYYY_MM.jsonl`

```json
{
  "timestamp": "2025-11-29T05:18:57Z",
  "location_id": "1-72662",
  "yr": {
    "kp_index": 3,
    "aurora_value": 0,
    "triggered": false
  }
}
```

### Workflow Runs
- Check Actions tab for execution history
- Failures upload diagnostic logs automatically

## 🐛 Troubleshooting

**"Not dark enough"**  
✓ Expected during daylight - bot exits quickly to save resources

**"Kp below threshold"**  
✓ Aurora activity too low - adjust `min_kp_index` lower if needed

**"YR.no location not found"**  
✗ Invalid `yr_location_id` - verify ID from YR.no URL

**"No webcams configured"**  
✗ Check `webcam_locations.json` exists and has valid URLs

**"AI rate limit"**  
✓ Temporary - bot will skip AI checks until quota resets

## 🔒 Privacy & Ethics

- Only processes public webcam feeds
- No personal data collected or stored
- Images deleted immediately after processing
- Respects API rate limits and terms of service
- Posts are clearly attributed to bot account

## 📜 License

MIT License - Feel free to fork and modify for your location!

## 🙏 Credits

- Aurora forecast data: [YR.no](https://www.yr.no/)
- Sun times: [sunrise-sunset.org](https://sunrise-sunset.org/)
- AI vision: Azure OpenAI GPT-4o
- Social platform: [Bluesky](https://bsky.app/)

---

**Lykke til med nordlys-jakten!** 🌌✨
