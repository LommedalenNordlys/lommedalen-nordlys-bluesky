#!/usr/bin/env bash
#
# Quick KP index check for aurora detection.
# Returns exit code 0 if KP >= threshold, 1 if below threshold.
# Uses NOAA Ovation Aurora model data.

set -euo pipefail

# Configuration
OVATION_URL="https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
TARGET_LAT=${1:-59}
TARGET_LON=${2:-10}
MIN_KP=${3:-4}

# Check if jq and curl are available
if ! command -v jq &> /dev/null; then
    echo "⚠️  jq not found; cannot check KP index" >&2
    exit 0  # Fail-open
fi

if ! command -v curl &> /dev/null; then
    echo "⚠️  curl not found; cannot check KP index" >&2
    exit 0  # Fail-open
fi

# Fetch Ovation data
echo "🌍 Fetching aurora activity data for coordinates ($TARGET_LAT, $TARGET_LON)..."

RESPONSE=$(curl -s --max-time 15 "$OVATION_URL" 2>/dev/null)

if [[ -z "$RESPONSE" ]]; then
    echo "⚠️  Failed to fetch Ovation data; assuming conditions are suitable (fail-open)" >&2
    exit 0
fi

# Extract coordinates array and find matching location
# The JSON structure is: {"coordinates": [[lat, lon, value], ...], ...}
# We look for exact match or nearest neighbor within 1 degree

KP_VALUE=$(echo "$RESPONSE" | jq -r --argjson lat "$TARGET_LAT" --argjson lon "$TARGET_LON" '
  .coordinates // [] 
  | map(select(
      (.[0] | floor) == ($lat | floor) and 
      (.[1] | floor) == ($lon | floor)
    ))
  | if length > 0 then .[0][2] else null end
')

# If no exact match, try nearest neighbor within 1 degree
if [[ "$KP_VALUE" == "null" ]] || [[ -z "$KP_VALUE" ]]; then
    KP_VALUE=$(echo "$RESPONSE" | jq -r --argjson lat "$TARGET_LAT" --argjson lon "$TARGET_LON" '
      .coordinates // []
      | map({
          lat: .[0],
          lon: .[1],
          value: .[2],
          dist: (($lat - .[0]) | fabs) + (($lon - .[1]) | fabs)
        })
      | map(select(.dist <= 1.0))
      | sort_by(.dist)
      | if length > 0 then .[0].value else null end
    ')
fi

if [[ "$KP_VALUE" == "null" ]] || [[ -z "$KP_VALUE" ]]; then
    echo "⚠️  No aurora data found for location ($TARGET_LAT, $TARGET_LON); assuming conditions are suitable (fail-open)" >&2
    exit 0
fi

# Compare with threshold
echo "📊 Local aurora intensity (Kp proxy): $KP_VALUE"
echo "🎯 Minimum required: $MIN_KP"

# Use bc for floating point comparison, fallback to integer comparison
if command -v bc &> /dev/null; then
    COMPARISON=$(echo "$KP_VALUE >= $MIN_KP" | bc -l)
    if [[ "$COMPARISON" == "1" ]]; then
        echo "✅ Aurora conditions favorable (Kp $KP_VALUE >= $MIN_KP)"
        exit 0
    else
        echo "🛑 Aurora activity too low (Kp $KP_VALUE < $MIN_KP)"
        exit 1
    fi
else
    # Fallback: multiply by 10 and compare as integers
    KP_INT=$(echo "$KP_VALUE * 10" | awk '{printf "%.0f", $1}')
    MIN_INT=$(echo "$MIN_KP * 10" | awk '{printf "%.0f", $1}')
    
    if [[ $KP_INT -ge $MIN_INT ]]; then
        echo "✅ Aurora conditions favorable (Kp $KP_VALUE >= $MIN_KP)"
        exit 0
    else
        echo "🛑 Aurora activity too low (Kp $KP_VALUE < $MIN_KP)"
        exit 1
    fi
fi
