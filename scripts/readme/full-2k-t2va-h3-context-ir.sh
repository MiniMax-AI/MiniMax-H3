#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

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
# Poll until prompt expansion completes, fails, or times out.
context_ir_result=$(pollMiniMaxTask "$MINIMAX_API_BASE" "$TOKEN" "$task_id")
echo "$context_ir_result" | jq .
# Export the complete expanded prompt for H3-Base and regeneration.
EXPANDED_PROMPT=$(echo "$context_ir_result" | jq -er '.task.content.prompt')
