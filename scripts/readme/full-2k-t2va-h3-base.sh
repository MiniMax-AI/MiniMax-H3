#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# Create the H3-Base request with the expanded prompt and capture the video ID.
video_id=$(
  jq -n \
    --arg prompt "$EXPANDED_PROMPT" \
    '{
  "task": "t2va",
  "prompt": $prompt,
  "conditions": [],
  "target": {
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "duration_seconds": 10
  },
  "seed": 0
}' |
    curl --silent --show-error \
      --request POST \
      --url "$SGLANG_DEPLOYMENT_URL/v1/videos" \
      --header 'Content-Type: application/json' \
      --data-binary @- |
    jq -er '.id'
)
# Poll until generation completes, fails, or times out.
pollSglangVideo "$SGLANG_DEPLOYMENT_URL" "$video_id" >/dev/null
# Download the local H3-Base MP4 after its status becomes completed.
curlWithRetry --location \
  --request GET \
  --url "$SGLANG_DEPLOYMENT_URL/v1/videos/$video_id/content" \
  --output t2va.mp4
