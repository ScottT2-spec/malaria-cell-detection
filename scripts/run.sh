#!/bin/bash
# Start MalariaAI backend server

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Load .env if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "🦟 Starting MalariaAI..."
echo "   Backend: http://localhost:${PORT:-8000}"
echo "   API docs: http://localhost:${PORT:-8000}/docs"
echo ""

cd backend
uvicorn app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --reload
