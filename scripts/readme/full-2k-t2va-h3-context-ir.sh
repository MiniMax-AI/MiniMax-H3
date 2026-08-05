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
      "text": "Epic space-opera theatrical teaser: a female captain stands alone before a massive observation window as the last fleet gathers and jumps away in a blinding flash, the bridge shaking, leaving her behind."
    }
  ],
  "duration": 10,
  "ratio": "16:9"
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
