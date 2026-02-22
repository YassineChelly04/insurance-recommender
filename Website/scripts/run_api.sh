#!/usr/bin/env bash
# InsureAI — Linux/macOS API Run Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Starting InsureAI Backend API..."
echo "Access: http://localhost:8000"
echo "Docs:   http://localhost:8000/docs"
echo ""

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
