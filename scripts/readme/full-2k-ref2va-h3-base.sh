#!/usr/bin/env bash
set -euo pipefail

# Create the H3-Base request with the expanded prompt and capture the video ID.
video_id=$(
  jq -n \
    --arg prompt "$EXPANDED_PROMPT" \
    '{
  "task": "ref2va",
  "prompt": $prompt,
  "conditions": [
    {
      "type": "video",
      "uri": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4",
      "role": "reference"
    },
    {
      "type": "audio",
      "uri": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/f463d523c5ce_01_%E9%9F%B3%E9%A2%911_RSLcbpzJPo_6%E6%9C%885%E6%97%A5(1).mp3",
      "role": "reference"
    }
  ],
  "target": {
    "short_edge": 768,
    "aspect_ratio": "auto",
    "duration_seconds": 5
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
# Query the generation status.
curl --silent --show-error \
  --request GET \
  --url "$SGLANG_DEPLOYMENT_URL/v1/videos/$video_id" |
  jq '{status}'
# Download the local H3-Base MP4 after its status becomes completed.
curl --silent --show-error \
  --request GET \
  --url "$SGLANG_DEPLOYMENT_URL/v1/videos/$video_id/content" \
  --output r2va.mp4
