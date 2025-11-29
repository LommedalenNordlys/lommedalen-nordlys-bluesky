#!/usr/bin/env bash
#
# Quick darkness check using cached sun schedule.
# Returns exit code 0 if dark, 1 if not dark.

set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-config.json}"

# Get schedule file from config or use default
if [[ -f "$CONFIG_FILE" ]] && command -v jq &>/dev/null; then
  SCHEDULE_FILE=$(jq -r '.data.sun_schedule_file // "sun_schedule.json"' "$CONFIG_FILE")
else
  SCHEDULE_FILE="sun_schedule.json"
fi

# Check if schedule file exists
if [[ ! -f "$SCHEDULE_FILE" ]]; then
    echo "⚠️  sun_schedule.json not found; assuming dark (fail-open)" >&2
    exit 0
fi

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "⚠️  jq not found; assuming dark (fail-open)" >&2
    exit 0
fi

# Get current UTC hour
CURRENT_HOUR=$(date -u +%H | sed 's/^0//')  # Remove leading zero

# Get dark hours from schedule
START_HOUR=$(jq -r '.dark_hours.start_hour' "$SCHEDULE_FILE")
END_HOUR=$(jq -r '.dark_hours.end_hour' "$SCHEDULE_FILE")
SPANS_MIDNIGHT=$(jq -r '.dark_hours.spans_midnight' "$SCHEDULE_FILE")

# Check if values are valid
if [[ "$START_HOUR" == "null" ]] || [[ "$END_HOUR" == "null" ]]; then
    echo "⚠️  Invalid sun_schedule.json; assuming dark (fail-open)" >&2
    exit 0
fi

# Check if it's dark based on hour ranges
IS_DARK=false

if [[ "$SPANS_MIDNIGHT" == "true" ]]; then
    # Dark hours span midnight (e.g., 17-23 and 0-4)
    if [[ $CURRENT_HOUR -ge $START_HOUR ]] || [[ $CURRENT_HOUR -le $END_HOUR ]]; then
        IS_DARK=true
    fi
else
    # Continuous dark period (rare, but possible in polar regions)
    if [[ $CURRENT_HOUR -ge $START_HOUR ]] && [[ $CURRENT_HOUR -le $END_HOUR ]]; then
        IS_DARK=true
    fi
fi

if [[ "$IS_DARK" == "true" ]]; then
    echo "✅ Dark (astronomical night)"
    exit 0
else
    echo "☀️  Not dark enough (astronomical twilight or daylight)"
    exit 1
fi
