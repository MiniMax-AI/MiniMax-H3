#!/usr/bin/env bash
set -euo pipefail

# Synchronize the canonical h3-prompt-writing skill (skills/h3-prompt-writing)
# to its agent install copies (.claude/skills and .agents/skills) so the three
# copies never drift apart. Run after editing the canonical SKILL.md,
# agents/openai.yaml, or references/ files, then commit the result.
#
# Usage:
#   ./scripts/sync-skill-copies.sh

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_skill="$repo_root/skills/h3-prompt-writing"

targets=(
  "$repo_root/.claude/skills/h3-prompt-writing"
  "$repo_root/.agents/skills/h3-prompt-writing"
)

for target in "${targets[@]}"; do
  mkdir -p "$target"
  cp -R "$source_skill/." "$target/"
  echo "Synced h3-prompt-writing -> $target"
done

echo "Done. Verify with: git diff --stat .claude/skills .agents/skills"
