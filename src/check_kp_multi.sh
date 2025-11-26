#!/usr/bin/env bash
#
# Multi-source KP/aurora check using both NOAA Ovation and YR.no APIs.
# Returns exit 0 if EITHER source indicates aurora activity, else 1.
# 
# NOAA: kp_max strictly > threshold
# YR.no: any current/upcoming auroraValue strictly > threshold (was >0; tightened per new policy)
#
# Usage: check_kp_multi.sh <lat> <lon> <min_kp> [radius] [yr_location_id]

set -euo pipefail

# Configuration
NOAA_URL="https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
YR_LOCATION_ID=${5:-"1-72662"}  # Default: Lommedalen
YR_URL="https://www.yr.no/api/v0/locations/${YR_LOCATION_ID}/auroraforecast?language=nb"

TARGET_LAT=${1:-59.959103}
TARGET_LON=${2:-10.4592208}
MIN_KP=${3:-3}
RADIUS=${4:-1.5}
KP_DATA_DIR="kp_data"

# Result trackers
NOAA_TRIGGERED=false
YR_TRIGGERED=false
NOAA_KP_MAX=0
YR_KP_INDEX=0
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
  
  printf "  📊 NOAA: samples=%d kp_max=%.2f kp_avg=%.2f threshold(>)=%.2f\n" "$SAMPLE_COUNT" "$NOAA_KP_MAX" "$NOAA_KP_AVG" "$MIN_KP"
  
  NOAA_MEETS=$(echo "$NOAA_KP_MAX > $MIN_KP" | bc -l || echo 0)
  if [[ "$NOAA_MEETS" == "1" ]]; then
    NOAA_TRIGGERED=true
    echo "  ✅ NOAA indicates aurora potential (kp_max $NOAA_KP_MAX > $MIN_KP)"
  else
    echo "  ❌ NOAA below strict threshold (kp_max $NOAA_KP_MAX <= $MIN_KP)"
  fi
else
  echo "  ⚠️ NOAA data unavailable"
fi

# ===== YR.no Aurora Forecast Check =====
echo "📡 Fetching YR.no aurora forecast for location ${YR_LOCATION_ID}"
YR_RESP=$(curl -s --max-time 15 "$YR_URL" 2>/dev/null || true)

if [[ -n "$YR_RESP" ]]; then
  # YR.no returns timestamps in local timezone (+01:00 for Norway)
  # Convert current UTC time to match YR's timezone for accurate interval matching
  NOW_LOCAL=$(date -u -d "1 hour" +"%Y-%m-%dT%H:%M:%S" 2>/dev/null || date -u -v+1H +"%Y-%m-%dT%H:%M:%S")
  
  # Find first interval that contains current time and extract kpIndex (primary) and auroraValue (secondary)
  YR_DATA=$(echo "$YR_RESP" | jq -r --arg now "$NOW_LOCAL" '
    .shortIntervals // []
    | map(select(.start <= ($now + "+01:00") and .end > ($now + "+01:00")))
    | first
    | if . then {kp: (.kpIndex // 0), aurora: (.auroraValue // 0)} else {kp: 0, aurora: 0} end
  ')
  
  YR_KP_INDEX=$(echo "$YR_DATA" | jq -r '.kp // 0')
  YR_AURORA_MAX=$(echo "$YR_DATA" | jq -r '.aurora // 0')
  
  printf "  📊 YR.no: kpIndex=%.0f auroraValue=%.2f (current interval) threshold(>)=%.2f\n" "$YR_KP_INDEX" "$YR_AURORA_MAX" "$MIN_KP"
  
  YR_MEETS=$(echo "$YR_KP_INDEX > $MIN_KP" | bc -l || echo 0)
  if [[ "$YR_MEETS" == "1" ]]; then
    YR_TRIGGERED=true
    echo "  ✅ YR.no indicates aurora potential (kpIndex $YR_KP_INDEX > $MIN_KP)"
  else
    echo "  ❌ YR.no below strict threshold (kpIndex $YR_KP_INDEX <= $MIN_KP)"
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
  --argjson yr_kp "${YR_KP_INDEX:-0}" \
  --argjson yr_aurora "$YR_AURORA_MAX" \
  --arg noaa_triggered "$NOAA_TRIGGERED" \
  --arg yr_triggered "$YR_TRIGGERED" \
  '{
    timestamp: $ts,
    target_lat: $lat,
    target_lon: $lon,
    sources: {
      noaa: {kp_max: $noaa_max, kp_avg: $noaa_avg, triggered: ($noaa_triggered == "true")},
      yr: {kp_index: $yr_kp, aurora_value: $yr_aurora, triggered: ($yr_triggered == "true")}
    }
  }'
)
echo "$LOG_LINE" >> "$FILE"
echo "💾 Saved multi-source log -> $FILE"

# ===== Final Decision (OR logic) =====
echo ""
if [[ "$NOAA_TRIGGERED" == "true" ]] || [[ "$YR_TRIGGERED" == "true" ]]; then
  echo "✅ Aurora conditions favorable (NOAA=${NOAA_TRIGGERED}, YR=${YR_TRIGGERED})"
  
  # Export KP values for Python script to use
  # Use the max value from triggered sources
  VALIDATED_KP=0
  if [[ "$YR_TRIGGERED" == "true" ]] && [[ "$NOAA_TRIGGERED" == "true" ]]; then
    # Both triggered: use max
    VALIDATED_KP=$(echo "if ($YR_KP_INDEX > $NOAA_KP_MAX) $YR_KP_INDEX else $NOAA_KP_MAX" | bc -l)
  elif [[ "$YR_TRIGGERED" == "true" ]]; then
    VALIDATED_KP=$YR_KP_INDEX
  elif [[ "$NOAA_TRIGGERED" == "true" ]]; then
    VALIDATED_KP=$NOAA_KP_MAX
  fi
  
  echo "VALIDATED_KP=$VALIDATED_KP"
  exit 0
else
  echo "🛑 No aurora activity detected from either source"
  exit 1
fi
