#!/usr/bin/env bash
set -euo pipefail

# Create the prompt-expansion task and capture its runtime ID.
task_id=$(
  curl --silent --show-error \
    --request POST \
    --url "$MINIMAX_API_BASE/v2/h3_context_ir" \
    --header "Authorization: Bearer $TOKEN" \
    --header 'Content-Type: application/json' \
    --data '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "Character speaks: Follow the wind, live free. Leave worries behind, enjoy the moment. Voice timbre follows reference audio 1."
    },
    {
      "type": "video_url",
      "video_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4"
      },
      "role": "reference_video"
    },
    {
      "type": "audio_url",
      "audio_url": {
        "url": "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/f463d523c5ce_01_%E9%9F%B3%E9%A2%911_RSLcbpzJPo_6%E6%9C%885%E6%97%A5(1).mp3"
      },
      "role": "reference_audio"
    }
  ],
  "duration": 5,
  "ratio": "adaptive"
}' |
    jq -er '.task_id'
)
# Query again while the task is queued or running.
context_ir_result=$(
  curl --silent --show-error \
    --request GET \
    --url "$MINIMAX_API_BASE/v2/query/video_generation/$task_id" \
    --header "Authorization: Bearer $TOKEN"
)
echo "$context_ir_result" | jq .
# Export the complete expanded prompt for H3-Base and regeneration.
EXPANDED_PROMPT=$(echo "$context_ir_result" | jq -er '.task.content.prompt')
