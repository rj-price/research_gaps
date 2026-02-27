#!/bin/bash

# Terminate background processes on exit
trap "kill 0" EXIT

echo "Starting Backend (FastAPI)..."
source .venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000 &

echo "Starting Frontend (Vite)..."
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173 &

# Keep script running
wait
