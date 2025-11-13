#!/usr/bin/env bash
#
# Multi-source KP/aurora check using both NOAA Ovation and YR.no APIs.
# Returns exit 0 if EITHER source indicates aurora activity, else 1.
# 
# NOAA: kp_max >= threshold
# YR.no: any current/upcoming auroraValue > 0
#
# Usage: check_kp_multi.sh <lat> <lon> <min_kp> [radius] [yr_location_id]

set -euo pipefail

# Configuration
NOAA_URL="https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
YR_LOCATION_ID=${5:-"1-72662"}  # Default: Lommedalen
YR_URL="https://www.yr.no/api/v0/locations/${YR_LOCATION_ID}/auroraforecast?language=nb"

TARGET_LAT=${1:-59.959103}
TARGET_LON=${2:-10.4592208}
MIN_KP=${3:-1}
RADIUS=${4:-1.5}
KP_DATA_DIR="kp_data"

# Result trackers
NOAA_TRIGGERED=false
YR_TRIGGERED=false
NOAA_KP_MAX=0
YR_AURORA_MAX=0

if ! command -v jq &>/dev/null; then
  echo "⚠️ jq not found; fail-open" >&2
  exit 0
fi
if ! command -v curl &>/dev/null; then
  echo "⚠️ curl not found; fail-open" >&2
  exit 0
fi

echo "🌍 Checking aurora conditions from NOAA and YR.no..."

# ===== NOAA Ovation Check =====
echo "📡 Fetching NOAA Ovation data for area around (${TARGET_LAT}, ${TARGET_LON}) radius=${RADIUS}°"
NOAA_RESP=$(curl -s --max-time 15 "$NOAA_URL" 2>/dev/null || true)

if [[ -n "$NOAA_RESP" ]]; then
  SAMPLES_JSON=$(echo "$NOAA_RESP" | jq -c --argjson lat "$TARGET_LAT" --argjson lon "$TARGET_LON" --argjson rad "$RADIUS" '
    (.coordinates // [])
    | map(select(((.[0]-$lat)|abs) <= $rad and ((.[1]-$lon)|abs) <= $rad))
    | map({lat: .[0], lon: .[1], value: .[2]})
  ')
  
  SAMPLE_COUNT=$(echo "$SAMPLES_JSON" | jq 'length')
  if [[ "$SAMPLE_COUNT" -eq 0 ]]; then
    echo "  ⚠️ No NOAA samples in radius; widening search"
    SAMPLES_JSON=$(echo "$NOAA_RESP" | jq -c --argjson lat "$TARGET_LAT" --argjson lon "$TARGET_LON" '
      (.coordinates // [])
      | map({lat: .[0], lon: .[1], value: .[2], dist: ((.[0]-$lat)|abs) + ((.[1]-$lon)|abs) })
      | sort_by(.dist)
      | .[:10]
      | map({lat, lon, value})
    ')
    SAMPLE_COUNT=$(echo "$SAMPLES_JSON" | jq 'length')
  fi
  
  NOAA_KP_MAX=$(echo "$SAMPLES_JSON" | jq '[.[].value] | max // 0')
  NOAA_KP_AVG=$(echo "$SAMPLES_JSON" | jq -r 'if length>0 then ([.[].value]|add/length) else 0 end')
  
  printf "  📊 NOAA: samples=%d kp_max=%.2f kp_avg=%.2f threshold=%.2f\n" "$SAMPLE_COUNT" "$NOAA_KP_MAX" "$NOAA_KP_AVG" "$MIN_KP"
  
  NOAA_MEETS=$(echo "$NOAA_KP_MAX >= $MIN_KP" | bc -l || echo 0)
  if [[ "$NOAA_MEETS" == "1" ]]; then
    NOAA_TRIGGERED=true
    echo "  ✅ NOAA indicates aurora potential (kp_max $NOAA_KP_MAX >= $MIN_KP)"
  else
    echo "  ❌ NOAA below threshold (kp_max $NOAA_KP_MAX < $MIN_KP)"
  fi
else
  echo "  ⚠️ NOAA data unavailable"
fi

# ===== YR.no Aurora Forecast Check =====
echo "📡 Fetching YR.no aurora forecast for location ${YR_LOCATION_ID}"
YR_RESP=$(curl -s --max-time 15 "$YR_URL" 2>/dev/null || true)

if [[ -n "$YR_RESP" ]]; then
  # Get current UTC time
  NOW_UTC=$(date -u +"%Y-%m-%dT%H:%M:%S")
  
  # Find intervals that overlap current time and extract max auroraValue
  YR_AURORA_MAX=$(echo "$YR_RESP" | jq -r --arg now "$NOW_UTC" '
    .shortIntervals // []
    | map(select(.start <= ($now + "+01:00") and .end > ($now + "+01:00")))
    | map(.auroraValue // 0)
    | max // 0
  ')
  
  # Also check next few hours (next 3 intervals = ~3 hours ahead)
  YR_UPCOMING_MAX=$(echo "$YR_RESP" | jq -r '
    .shortIntervals // []
    | .[0:3]
    | map(.auroraValue // 0)
    | max // 0
  ')
  
  # Use the higher of current or upcoming
  if (( $(echo "$YR_UPCOMING_MAX > $YR_AURORA_MAX" | bc -l) )); then
    YR_AURORA_MAX=$YR_UPCOMING_MAX
  fi
  
  printf "  📊 YR.no: max auroraValue=%.2f (current/next 3hrs)\n" "$YR_AURORA_MAX"
  
  YR_MEETS=$(echo "$YR_AURORA_MAX > 0" | bc -l || echo 0)
  if [[ "$YR_MEETS" == "1" ]]; then
    YR_TRIGGERED=true
    echo "  ✅ YR.no indicates aurora potential (auroraValue $YR_AURORA_MAX > 0)"
  else
    echo "  ❌ YR.no shows no activity (auroraValue $YR_AURORA_MAX = 0)"
  fi
else
  echo "  ⚠️ YR.no data unavailable"
fi

# ===== Persist Log =====
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
YEAR=$(date -u +"%Y")
MONTH=$(date -u +"%m")
DIR="${KP_DATA_DIR}/${YEAR}/${MONTH}"
mkdir -p "$DIR"
FILE="${DIR}/${YEAR}_${MONTH}.jsonl"

LOG_LINE=$(jq -n \
  --arg ts "$TS" \
  --argjson lat "$TARGET_LAT" \
  --argjson lon "$TARGET_LON" \
  --argjson noaa_max "$NOAA_KP_MAX" \
  --argjson noaa_avg "${NOAA_KP_AVG:-0}" \
  --argjson yr_max "$YR_AURORA_MAX" \
  --arg noaa_triggered "$NOAA_TRIGGERED" \
  --arg yr_triggered "$YR_TRIGGERED" \
  '{
    timestamp: $ts,
    target_lat: $lat,
    target_lon: $lon,
    sources: {
      noaa: {kp_max: $noaa_max, kp_avg: $noaa_avg, triggered: ($noaa_triggered == "true")},
      yr: {aurora_max: $yr_max, triggered: ($yr_triggered == "true")}
    }
  }'
)
echo "$LOG_LINE" >> "$FILE"
echo "💾 Saved multi-source log -> $FILE"

# ===== Final Decision (OR logic) =====
echo ""
if [[ "$NOAA_TRIGGERED" == "true" ]] || [[ "$YR_TRIGGERED" == "true" ]]; then
  echo "✅ Aurora conditions favorable (NOAA=${NOAA_TRIGGERED}, YR=${YR_TRIGGERED})"
  exit 0
else
  echo "🛑 No aurora activity detected from either source"
  exit 1
fi
