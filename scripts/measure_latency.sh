#!/bin/bash

poetry run uvicorn api.main:app --port 8000 &
SERVER_PID=$!
sleep 45    # wait for model to load

# Time a real request
TIME_START=$(date +%s%N)
curl -s -X POST http://localhost:8000/detect \
  -F "file=@data/annotated/images/test/0bf13796-img_0789.jpg" \
  -o /home/islam_tb/Documents/AlgerPlate/tmp/result.json
TIME_END=$(date +%s%N)
MS=$(( (TIME_END - TIME_START) / 1000000 ))
echo "Latency: ${MS}ms"
cat /home/islam_tb/Documents/AlgerPlate/tmp/result.json | python3 -m json.tool | grep latency_ms

kill $SERVER_PID