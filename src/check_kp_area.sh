#!/usr/bin/env bash
# Multi-point KP index check using NOAA Ovation model.
# Samples all coordinate triples within a radius window around a precise target location
# and aggregates their values. Exits 0 if kp_max >= threshold, else 1.
# Usage: check_kp_area.sh <lat> <lon> <min_kp> [radius]

set -euo pipefail

OVATION_URL="https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
TARGET_LAT=${1:-59.959103}
TARGET_LON=${2:-10.4592208}
MIN_KP=${3:-1}
RADIUS=${4:-1.5}   # degrees (lat/lon box half-width)
KP_DATA_DIR="kp_data"

if ! command -v jq &>/dev/null; then
  echo "⚠️ jq not found; fail-open" >&2
  exit 0
fi
if ! command -v curl &>/dev/null; then
  echo "⚠️ curl not found; fail-open" >&2
  exit 0
fi

echo "🌍 Fetching Ovation data for area around (${TARGET_LAT}, ${TARGET_LON}) radius=${RADIUS}° ..."
RESP=$(curl -s --max-time 15 "$OVATION_URL" 2>/dev/null || true)
if [[ -z "$RESP" ]]; then
  echo "⚠️ Empty response; fail-open" >&2
  exit 0
fi

# Extract samples within radius
SAMPLES_JSON=$(echo "$RESP" | jq -c --argjson lat "$TARGET_LAT" --argjson lon "$TARGET_LON" --argjson rad "$RADIUS" '
  (.coordinates // [])
  | map(select(((.[0]-$lat)|abs) <= $rad and ((.[1]-$lon)|abs) <= $rad))
  | map({lat: .[0], lon: .[1], value: .[2]})
')

SAMPLE_COUNT=$(echo "$SAMPLES_JSON" | jq 'length')
if [[ "$SAMPLE_COUNT" -eq 0 ]]; then
  echo "⚠️ No samples in radius; widening search to 1.0°" >&2
  SAMPLES_JSON=$(echo "$RESP" | jq -c --argjson lat "$TARGET_LAT" --argjson lon "$TARGET_LON" '
    (.coordinates // [])
    | map({lat: .[0], lon: .[1], value: .[2], dist: ((.[0]-$lat)|abs) + ((.[1]-$lon)|abs) })
    | sort_by(.dist)
    | .[:10]  # take nearest 10 as fallback
    | map({lat, lon, value})
  ')
  SAMPLE_COUNT=$(echo "$SAMPLES_JSON" | jq 'length')
fi

KP_MAX=$(echo "$SAMPLES_JSON" | jq '[.[].value] | max // 0')
KP_AVG=$(echo "$SAMPLES_JSON" | jq -r 'if length>0 then ([.[].value]|add/length) else 0 end')

printf "📊 KP samples: count=%d max=%.2f avg=%.2f threshold=%.2f\n" "$SAMPLE_COUNT" "$KP_MAX" "$KP_AVG" "$MIN_KP"

# Persist log (JSONL)
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
YEAR=$(date -u +"%Y")
MONTH=$(date -u +"%m")
DIR="${KP_DATA_DIR}/${YEAR}/${MONTH}"
mkdir -p "$DIR"
FILE="${DIR}/${YEAR}_${MONTH}.jsonl"
# Build one line of JSON containing samples array
LOG_LINE=$(jq -n --arg ts "$TS" --argjson lat "$TARGET_LAT" --argjson lon "$TARGET_LON" --argjson min "$MIN_KP" --argjson max "$KP_MAX" --argjson avg "$KP_AVG" --argjson samples "$SAMPLES_JSON" '{timestamp:$ts, target_lat:$lat, target_lon:$lon, kp_threshold:$min, kp_max:$max, kp_avg:$avg, samples:$samples}')
echo "$LOG_LINE" >> "$FILE"

echo "💾 Saved multi-point KP log -> $FILE"

# Decision
# Use max value for threshold gating
MEETS=$(echo "$KP_MAX >= $MIN_KP" | bc -l || echo 0)
if [[ "$MEETS" == "1" ]]; then
  echo "✅ Aurora conditions potentially favorable (kp_max $KP_MAX >= $MIN_KP)"
  exit 0
else
  echo "🛑 Aurora activity below threshold (kp_max $KP_MAX < $MIN_KP)"
  exit 1
fi
