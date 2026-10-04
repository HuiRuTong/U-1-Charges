#!/bin/bash

# This script compiles everything inside ./checks/

set -e
curr_dir="$(cd "$(dirname "$0")" && pwd)"

cd $curr_dir/../checks

if [ ! -d "../bin" ]; then
    mkdir ../bin/
fi

gcc clean_sol.c UTILS.c -o ../bin/clean_sol
gcc conditions_2_electric_boogaloo.c UTILS.c -o ../bin/check_conditions
gcc check_duplicates.c UTILS.c -o ../bin/check_duplicates
gcc sort_sol.c UTILS.c -o ../bin/sort_sol

echo "Finished compiling"