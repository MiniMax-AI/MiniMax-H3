#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# Create a 10-second 2K video directly and capture its runtime task ID.
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
      "text": "Epic space-opera theatrical teaser: a female captain stands alone before a massive observation window as the last fleet gathers and jumps away in a blinding flash, the bridge shaking, leaving her behind."
    }
  ],
  "resolution": "2K",
  "duration": 10,
  "ratio": "16:9"
}' |
    jq -er '.task_id'
)
# Poll until generation completes, fails, or times out.
generation_result=$(pollMiniMaxTask "$MINIMAX_API_BASE" "$TOKEN" "$task_id")
echo "$generation_result" | jq '{status: .task.status}'
# Download the 2K MP4 after the task succeeds.
video_url=$(echo "$generation_result" | jq -er '.task.content.url')
curlWithRetry --location "$video_url" --output h3_direct_2k.mp4
