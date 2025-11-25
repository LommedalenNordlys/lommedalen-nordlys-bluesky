#!/usr/bin/env bash
# Nordkapp multi-source aurora check (NOAA + YR.no) logging to kp_data_nordkapp
# Usage: check_kp_multi_nordkapp.sh [min_kp] [radius]
set -euo pipefail
NOAA_URL="https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
YR_LOCATION_ID="1-2403657"  # Nordkapp
YR_URL="https://www.yr.no/api/v0/locations/${YR_LOCATION_ID}/auroraforecast?language=nb"
TARGET_LAT=70.977376
TARGET_LON=24.6618412
MIN_KP=${1:-1}
RADIUS=${2:-1.5}
KP_DATA_DIR="kp_data_nordkapp"
NOAA_TRIGGERED=false
YR_TRIGGERED=false
NOAA_KP_MAX=0
YR_KP_INDEX=0
YR_AURORA_MAX=0
if ! command -v jq &>/dev/null; then echo "jq missing"; exit 0; fi
if ! command -v curl &>/dev/null; then echo "curl missing"; exit 0; fi
echo "🧪 Nordkapp: Checking aurora (NOAA + YR.no)";
# NOAA
RESP=$(curl -s --max-time 15 "$NOAA_URL" || true)
if [[ -n "$RESP" ]]; then
  SAMPLES=$(echo "$RESP" | jq -c --argjson lat $TARGET_LAT --argjson lon $TARGET_LON --argjson rad $RADIUS '(.coordinates // []) | map(select(((.[0]-$lat)|abs) <= $rad and ((.[1]-$lon)|abs) <= $rad)) | map({lat: .[0], lon: .[1], value: .[2]})')
  COUNT=$(echo "$SAMPLES" | jq 'length')
  if [[ $COUNT -eq 0 ]]; then
    SAMPLES=$(echo "$RESP" | jq -c --argjson lat $TARGET_LAT --argjson lon $TARGET_LON '(.coordinates // []) | map({lat: .[0], lon: .[1], value: .[2], dist: ((.[0]-$lat)|abs)+((.[1]-$lon)|abs) }) | sort_by(.dist) | .[:12] | map({lat,lon,value})')
    COUNT=$(echo "$SAMPLES" | jq 'length')
  fi
  NOAA_KP_MAX=$(echo "$SAMPLES" | jq '[.[].value] | max // 0')
  NOAA_KP_AVG=$(echo "$SAMPLES" | jq 'if length>0 then ([.[].value]|add/length) else 0 end')
  if (( $(echo "$NOAA_KP_MAX > $MIN_KP" | bc -l) )); then NOAA_TRIGGERED=true; fi
  echo "  NOAA kp_max=$NOAA_KP_MAX avg=$NOAA_KP_AVG samples=$COUNT threshold(>)=$MIN_KP"
else
  echo "  NOAA unavailable"
fi
# YR.no
YR_RESP=$(curl -s --max-time 15 "$YR_URL" || true)
if [[ -n "$YR_RESP" ]]; then
  NOW=$(date -u +"%Y-%m-%dT%H:%M:%S")
  YR_DATA=$(echo "$YR_RESP" | jq -r --arg now "$NOW" '(.shortIntervals // []) | map(select(.start <= ($now+"+01:00") and .end > ($now+"+01:00"))) | first | if . then {kp: (.kpIndex // 0), aurora: (.auroraValue // 0)} else {kp: 0, aurora: 0} end')
  YR_KP_INDEX=$(echo "$YR_DATA" | jq -r '.kp // 0')
  YR_AURORA_MAX=$(echo "$YR_DATA" | jq -r '.aurora // 0')
  if (( $(echo "$YR_KP_INDEX > $MIN_KP" | bc -l) )); then YR_TRIGGERED=true; fi
  echo "  YR kpIndex=$YR_KP_INDEX auroraValue=$YR_AURORA_MAX (current) threshold(>)=$MIN_KP"
else
  echo "  YR.no unavailable"
fi
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
YEAR=$(date -u +"%Y")
MONTH=$(date -u +"%m")
DIR="$KP_DATA_DIR/$YEAR/$MONTH"; mkdir -p "$DIR"; FILE="$DIR/${YEAR}_${MONTH}.jsonl"
LINE=$(jq -n --arg ts "$TS" --argjson lat $TARGET_LAT --argjson lon $TARGET_LON --argjson noaa_max $NOAA_KP_MAX --argjson noaa_avg ${NOAA_KP_AVG:-0} --argjson yr_kp ${YR_KP_INDEX:-0} --argjson yr_aurora $YR_AURORA_MAX --arg noaa_trig "$NOAA_TRIGGERED" --arg yr_trig "$YR_TRIGGERED" '{timestamp:$ts, target_lat:$lat, target_lon:$lon, sources:{noaa:{kp_max:$noaa_max,kp_avg:$noaa_avg,triggered:($noaa_trig=="true")}, yr:{kp_index:$yr_kp,aurora_value:$yr_aurora,triggered:($yr_trig=="true")}}}' )
echo "$LINE" >> "$FILE"
echo "  Logged -> $FILE"
if [[ "$NOAA_TRIGGERED" == "true" || "$YR_TRIGGERED" == "true" ]]; then echo "✅ Nordkapp aurora potential"; exit 0; else echo "🛑 Nordkapp no activity"; exit 1; fi
