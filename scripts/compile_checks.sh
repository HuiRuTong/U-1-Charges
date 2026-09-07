#!/bin/bash

# This script compiles everything inside ./checks/

set -e
curr_dir="$(cd "$(dirname "$0")" && pwd)"

cd $curr_dir/../checks

mkdir ../bin/

gcc check_duplicates.c UTILS.c -o ../bin/dupe
gcc sort_sol.c UTILS.c -o ../bin/sort
gcc rm_oob_sol.c UTILS.c -o ../bin/rm
gcc conditions_2_electric_boogaloo.c UTILS.c -o ../bin/cond

echo "finished compiling"