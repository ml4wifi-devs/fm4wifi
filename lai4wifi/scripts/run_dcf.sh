#!/bin/bash

cd "$(dirname "$0")/../.." && mkdir -p results && cd results

for IDX in 0 1 2 3 4 5 6 7; do
    python -m lai4wifi.baselines.dcf --scenarios residential --idx $IDX
done
