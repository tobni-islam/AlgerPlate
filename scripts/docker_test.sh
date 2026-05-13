#!/bin/bash

docker compose up -d

# All three endpoints from the host machine
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl -X POST http://localhost:8000/detect \
  -F "file=@data/annotated/images/test/0bf13796-img_0789.jpg" \
  | python3 -m json.tool

docker compose down