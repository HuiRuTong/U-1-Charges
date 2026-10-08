#!/bin/bash

# This script runs main.py 5 separate times with seeds 0, 1, 2, 3, 4

set -e
curr_dir="$(cd "$(dirname "$0")" && pwd)"

read -p "Identifier: " identifier

for i in {0..5}; do
    echo "Running trial $i"
    python3 "$curr_dir/..//main.py" "$i" "$curr_dir/../found_charges/$identifier _$i"
    echo "Finished trial $i"
    echo
done