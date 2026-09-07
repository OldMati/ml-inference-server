#!/usr/bin/env bash
# sweep.sh <policy>   -- server must already be running with that policy
POLICY=$1
OUT=results/$POLICY
mkdir -p "$OUT"

for rate in 20 50 100 200 300 400 500 600 700 800 900; do
    echo "=== $POLICY rate=$rate ==="
    ./loadgen/loadgen $rate 128 60 127.0.0.1 8080 fixtures/sample_input.bin "$OUT/rate_${rate}.csv"
    sleep 3
done
