#!/usr/bin/env bash
# Darkness check for Nordkapp using sun_schedule_nordkapp.json
# Exit 0 if dark, 1 if not dark.
set -euo pipefail
SCHEDULE_FILE="sun_schedule_nordkapp.json"
if [[ ! -f "$SCHEDULE_FILE" ]]; then echo "sun_schedule_nordkapp.json missing; assume dark"; exit 0; fi
if ! command -v jq &>/dev/null; then echo "jq missing; assume dark"; exit 0; fi
CUR_HOUR=$(date -u +%H | sed 's/^0//')
START=$(jq -r '.dark_hours.start_hour' "$SCHEDULE_FILE")
END=$(jq -r '.dark_hours.end_hour' "$SCHEDULE_FILE")
SPAN=$(jq -r '.dark_hours.spans_midnight' "$SCHEDULE_FILE")
if [[ "$START" == "null" || "$END" == "null" ]]; then echo "Invalid schedule; assume dark"; exit 0; fi
IS_DARK=false
if [[ "$SPAN" == "true" ]]; then
  if [[ $CUR_HOUR -ge $START || $CUR_HOUR -le $END ]]; then IS_DARK=true; fi
else
  if [[ $CUR_HOUR -ge $START && $CUR_HOUR -le $END ]]; then IS_DARK=true; fi
fi
if [[ "$IS_DARK" == "true" ]]; then echo "✅ Nordkapp dark"; exit 0; else echo "☀️ Nordkapp not dark"; exit 1; fi
