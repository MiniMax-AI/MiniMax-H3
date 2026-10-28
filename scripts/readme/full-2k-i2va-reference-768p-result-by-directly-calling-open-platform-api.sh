#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# Create an 8-second 768P FL2VA video directly and capture its runtime task ID.
task_id=$(
  curl --silent --show-error \
    --request POST \
    --url "$MINIMAX_API_BASE/v2/video_generation" \
    --header "Authorization: Bearer $TOKEN" \
    --header 'Content-Type: application/json' \
    --data '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "Pull focus to the people in the background and add more steam to the ramen bowl."
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/H3_AA_I2VA/gallery/sr_v17_variants_seed42_43_20260724/inputs/4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png"
      },
      "role": "first_frame"
    }
  ],
  "resolution": "768P",
  "duration": 8,
  "ratio": "adaptive"
}' |
    jq -er '.task_id'
)
# Poll until generation completes, fails, or times out.
generation_result=$(pollMiniMaxTask "$MINIMAX_API_BASE" "$TOKEN" "$task_id")
echo "$generation_result" | jq '{status: .task.status}'
# Download the 768P MP4 after the task succeeds.
video_url=$(echo "$generation_result" | jq -er '.task.content.url')
curlWithRetry --location "$video_url" --output i2va_direct_768p.mp4
