#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -f .env ]]; then
  echo "Deployment stopped: /opt/benchmind/.env is missing." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Deployment stopped: tracked files have local changes." >&2
  exit 1
fi

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "main" ]]; then
  echo "Deployment stopped: expected branch main, found $current_branch." >&2
  exit 1
fi

exec 9>/tmp/benchmind-deploy.lock
if ! flock -n 9; then
  echo "Deployment stopped: another deployment is running." >&2
  exit 1
fi

echo "Fetching the latest main branch..."
git fetch --prune origin main
git merge --ff-only origin/main

echo "Building and starting BenchMind..."
mkdir -p .benchmind
docker compose config --quiet
docker compose up -d --build --remove-orphans --wait --wait-timeout 180

echo "Deployment complete: $(git rev-parse --short HEAD)"
docker compose ps
