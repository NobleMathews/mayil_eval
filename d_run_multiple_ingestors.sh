#!/bin/bash

# Number of parallel processes
N=4

SCRIPT_PATH="c_setup_mayil_ingestion.py"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

for i in $(seq 0 $((N - 1)))
do
    python "$SCRIPT_PATH" $N $i 2>"$LOG_DIR/log_$i.log" &
done

wait
echo "All processes have completed."
