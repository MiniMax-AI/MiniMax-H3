#!/usr/bin/env bash
set -euo pipefail

H3_BASE_VIDEO='./t2va.mp4'
# Encode the local H3-Base video as a Data URL and create the regeneration task.
task_id=$(
  jq -n \
    --arg prompt "$EXPANDED_PROMPT" \
    --rawfile base_video <(
      printf 'data:video/mp4;base64,'
      base64 -w0 "$H3_BASE_VIDEO"
    ) \
    '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": $prompt
    },
    {
      "type": "video_url",
      "video_url": {
        "url": $base_video
      },
      "role": "base_video"
    }
  ],
  "resolution": "2K"
}' |
    curl --silent --show-error \
      --request POST \
      --url "$MINIMAX_API_BASE/v2/video_regeneration" \
      --header "Authorization: Bearer $TOKEN" \
      --header 'Content-Type: application/json' \
      --data-binary @- |
    jq -er '.task_id'
)
# Query again while the task is queued or running.
regeneration_result=$(
  curl --silent --show-error \
    --request GET \
    --url "$MINIMAX_API_BASE/v2/query/video_generation/$task_id" \
    --header "Authorization: Bearer $TOKEN"
)
echo "$regeneration_result" | jq '{status: .task.status}'
# Download the 2K MP4 after the task succeeds.
video_url=$(echo "$regeneration_result" | jq -er '.task.content.url')
curl --location "$video_url" --output t2va_2k.mp4
