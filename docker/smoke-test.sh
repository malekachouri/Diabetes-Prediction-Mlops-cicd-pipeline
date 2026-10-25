#!/usr/bin/env bash
# Smoke test of the API image: start a container, wait for the API, then send
# a few real requests. Usage: docker/smoke-test.sh IMAGE
set -euo pipefail

image="${1:?usage: $0 IMAGE}"
port="${SMOKE_PORT:-8099}"
name="diabetes-api-smoke-$$"
url="http://127.0.0.1:${port}"
patient='{"pregnancies": 2, "glucose": 148, "blood_pressure": 72, "bmi": 33.6, "age": 50}'

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "Smoke test FAILED, container logs:"
    docker logs "$name" 2>&1 | tail -n 30 || true
  fi
  docker rm -f "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "$name" -p "127.0.0.1:${port}:8000" "$image" >/dev/null

echo "Waiting for the API to start..."
for _ in $(seq 1 30); do
  if curl -fsS "$url/health" >/dev/null 2>&1; then
    break
  fi
  if [ "$(docker inspect --format '{{.State.Running}}' "$name")" != "true" ]; then
    echo "The container stopped before the API was ready"
    exit 1
  fi
  sleep 1
done

echo "GET /health   -> $(curl -fsS "$url/health")"

prediction="$(curl -fsS -X POST "$url/predict" -H 'Content-Type: application/json' -d "$patient")"
echo "POST /predict -> $prediction"
grep -q '"probability"' <<<"$prediction"

code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$url/predict" \
  -H 'Content-Type: application/json' -d '{"pregnancies": 2, "glucose": 0, "age": 50}')"
echo "POST /predict with glucose=0 -> HTTP $code"
[ "$code" = "422" ]

curl -fsS "$url/metrics" | grep -q '^diabetes_api_predictions_total{diabetic='
echo "GET /metrics  -> prediction counters exposed"

uid="$(docker exec "$name" id -u)"
echo "Container user id -> $uid"
[ "$uid" != "0" ]

echo "Smoke test passed"
