#!/bin/bash

cd "$(dirname "$0")/../.." && mkdir -p results && cd results

for SCENARIOS in residential enterprise; do
    for IDX in 0 1 2 3 4 5 6 7; do
        python -m lai4wifi.baselines.static --scenarios $SCENARIOS --idx $IDX --output baseline_${SCENARIOS}_${IDX}.json
        python -m lai4wifi.baselines.analyze --scenarios $SCENARIOS --idx $IDX -i baseline_${SCENARIOS}_${IDX}.json -o baseline_${SCENARIOS}_${IDX}_analysis.json
    done
done

python -m lai4wifi.baselines.static --scenarios random --output baseline_random.json
python -m lai4wifi.baselines.analyze --scenarios random -i baseline_random.json -o baseline_random_analysis.json
