#!/bin/bash

# Terminate background processes when the script exits
trap 'kill $(jobs -p) 2>/dev/null' EXIT

echo "Starting Python FastAPI engine..."
python ai_engine/main.py &
PYTHON_PID=$!

echo "Starting Node.js Express backend..."
node backend/server.js &
NODE_PID=$!

echo "Both processes started. Monitoring..."
# wait -n returns when any background job exits
wait -n $PYTHON_PID $NODE_PID

echo "One of the services has exited. Terminating..."
