#!/usr/bin/env bash
# Downloads a YouTube video's audio (mono, 16kHz WAV) and splits it into
# fixed-length chunks for the multi-label co-occurrence EDA (Step 3).
#
# Usage:
#   scripts/fetch_multilabel_audio.sh <youtube_url> [segment_seconds]
#
# Example:
#   scripts/fetch_multilabel_audio.sh "https://youtube.com/watch?v=XXXX" 300

set -euo pipefail

URL="${1:?Usage: fetch_multilabel_audio.sh <youtube_url> [segment_seconds]}"
SEGMENT_SECONDS="${2:-300}"

OUT_DIR="eda_outputs/multilabel_sources"
mkdir -p "$OUT_DIR"

echo "Downloading full audio..."
yt-dlp -x --audio-format wav \
  --postprocessor-args "-ar 16000 -ac 1" \
  --restrict-filenames \
  -o "${OUT_DIR}/%(title).60s.%(ext)s" \
  "$URL"

# --get-filename predicts the pre-extraction name (wrong extension), so find
# the actual .wav that was just written instead of asking yt-dlp to guess again.
FULL_FILE=$(ls -t "${OUT_DIR}"/*.wav 2>/dev/null | grep -v '_part_' | head -1)
STEM=$(basename "$FULL_FILE" .wav)

echo "Splitting into ${SEGMENT_SECONDS}s chunks..."
ffmpeg -y -i "$FULL_FILE" \
  -f segment -segment_time "$SEGMENT_SECONDS" -c copy \
  "${OUT_DIR}/${STEM}_part_%02d.wav"

NUM_PARTS=$(ls "${OUT_DIR}/${STEM}_part_"*.wav | wc -l | tr -d ' ')

SOURCES_CSV="${OUT_DIR}/sources.csv"
if [ ! -f "$SOURCES_CSV" ]; then
  echo "full_file,youtube_url,segment_seconds,num_parts,downloaded_at,license_notes" > "$SOURCES_CSV"
fi
echo "${FULL_FILE},${URL},${SEGMENT_SECONDS},${NUM_PARTS},$(date -u +%Y-%m-%dT%H:%M:%SZ)," >> "$SOURCES_CSV"

echo "Done. Full file: $FULL_FILE"
echo "Parts: ${NUM_PARTS} chunk(s) at ${OUT_DIR}/${STEM}_part_*.wav"
echo "Logged to $SOURCES_CSV -- fill in the license_notes column by hand."
