#!/usr/bin/env bash
#
# KP/aurora check using YR.no API.
# Returns exit 0 if YR.no indicates aurora activity (kpIndex > threshold), else 1.
#
# Usage: check_kp_multi.sh [min_kp] [yr_location_id]

set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-config.json}"

# Load configuration
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "⚠️ config.json not found; using defaults" >&2
fi

# Read from config file or use arguments/defaults
if [[ -f "$CONFIG_FILE" ]] && command -v jq &>/dev/null; then
  DEFAULT_MIN_KP=$(jq -r '.aurora.min_kp_index // 3' "$CONFIG_FILE")
  DEFAULT_LOCATION_ID=$(jq -r '.location.yr_location_id // "1-72662"' "$CONFIG_FILE")
  KP_DATA_DIR=$(jq -r '.data.kp_log_dir // "kp_data"' "$CONFIG_FILE")
  YR_URL_TEMPLATE=$(jq -r '.api.yr_aurora_forecast_url // "https://www.yr.no/api/v0/locations/{location_id}/auroraforecast?language=nb"' "$CONFIG_FILE")
else
  DEFAULT_MIN_KP=3
  DEFAULT_LOCATION_ID="1-72662"
  KP_DATA_DIR="kp_data"
  YR_URL_TEMPLATE="https://www.yr.no/api/v0/locations/{location_id}/auroraforecast?language=nb"
fi

# Configuration (args override config file)
MIN_KP=${1:-$DEFAULT_MIN_KP}
YR_LOCATION_ID=${2:-$DEFAULT_LOCATION_ID}
YR_URL="${YR_URL_TEMPLATE/\{location_id\}/$YR_LOCATION_ID}"

# Result trackers
YR_TRIGGERED=false
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

echo "🌍 Checking aurora conditions from YR.no..."

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
    echo "  ❌ YR.no below threshold (kpIndex $YR_KP_INDEX <= $MIN_KP)"
  fi
else
  echo "  ⚠️ YR.no data unavailable; fail-open"
  exit 0
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
  --arg location_id "$YR_LOCATION_ID" \
  --argjson yr_kp "${YR_KP_INDEX:-0}" \
  --argjson yr_aurora "$YR_AURORA_MAX" \
  --arg yr_triggered "$YR_TRIGGERED" \
  '{
    timestamp: $ts,
    location_id: $location_id,
    yr: {kp_index: $yr_kp, aurora_value: $yr_aurora, triggered: ($yr_triggered == "true")}
  }'
)
echo "$LOG_LINE" >> "$FILE"
echo "💾 Saved YR.no log -> $FILE"

# ===== Final Decision =====
echo ""
if [[ "$YR_TRIGGERED" == "true" ]]; then
  echo "✅ Aurora conditions favorable (YR.no kpIndex: $YR_KP_INDEX > $MIN_KP)"
  echo "VALIDATED_KP=$YR_KP_INDEX"
  exit 0
else
  echo "🛑 No aurora activity detected (YR.no kpIndex: $YR_KP_INDEX <= $MIN_KP)"
  exit 1
fi
