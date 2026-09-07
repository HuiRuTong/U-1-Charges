#!/bin/bash

# This script builds the C extensions inside the aptly named C folder

set -e
curr_dir="$(cd "$(dirname "$0")" && pwd)"

cd $curr_dir/../C

python3 build.py build_ext --inplace

echo "finished building C extensions"