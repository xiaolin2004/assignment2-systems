#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-ali-heyuan}"
REMOTE_REPO_PATH="${REMOTE_REPO_PATH:-/root/assignment2-systems}"
BRANCH="${BRANCH:-$(git -C "$REPO_ROOT" branch --show-current)}"
COMMIT_MESSAGE="${1:-${COMMIT_MESSAGE:-sync from $(hostname -s) at $(date -u '+%Y-%m-%d %H:%M:%S UTC')}}"

if [[ -z "$BRANCH" ]]; then
  echo "Unable to determine git branch." >&2
  exit 1
fi

for cmd in git rsync ssh; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

if ! git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Repository root is not a git work tree: $REPO_ROOT" >&2
  exit 1
fi

REMOTE_COMMIT_MESSAGE="$(printf '%q' "$COMMIT_MESSAGE")"

echo "Syncing $REPO_ROOT to $REMOTE_HOST:$REMOTE_REPO_PATH"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_REPO_PATH'"
rsync -az --delete "$REPO_ROOT"/ "$REMOTE_HOST:$REMOTE_REPO_PATH/"

echo "Configuring remote git safety checks"
ssh "$REMOTE_HOST" "git config --global --add safe.directory '$REMOTE_REPO_PATH'"

echo "Remote git status"
REMOTE_STATUS="$(ssh "$REMOTE_HOST" "git -C '$REMOTE_REPO_PATH' status --short")"
if [[ -n "$REMOTE_STATUS" ]]; then
  printf '%s\n' "$REMOTE_STATUS"
  echo "Creating remote commit"
  ssh "$REMOTE_HOST" "git -C '$REMOTE_REPO_PATH' add -A"
  ssh "$REMOTE_HOST" "git -C '$REMOTE_REPO_PATH' commit -m $REMOTE_COMMIT_MESSAGE"
else
  echo "Remote work tree clean; no new commit created."
fi

echo "Pushing $BRANCH from $REMOTE_HOST to GitHub"
ssh "$REMOTE_HOST" "git -C '$REMOTE_REPO_PATH' push origin '$BRANCH'"

echo "Done."
