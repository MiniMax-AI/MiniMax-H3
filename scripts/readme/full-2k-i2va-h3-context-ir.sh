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
  "duration": 8,
  "ratio": "adaptive"
}' |
    jq -er '.task_id'
)
# Poll until prompt expansion completes, fails, or times out.
context_ir_result=$(pollMiniMaxTask "$MINIMAX_API_BASE" "$TOKEN" "$task_id")
echo "$context_ir_result" | jq .
# Export the complete expanded prompt for H3-Base and regeneration.
EXPANDED_PROMPT=$(echo "$context_ir_result" | jq -er '.task.content.prompt')
