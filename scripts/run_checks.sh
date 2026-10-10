#!/bin/bash

# This script cleans up the solutions to remove any same-file duplicates
# Then cross compares with another file

set -e
curr_dir="$(cd "$(dirname "$0")" && pwd)"

read -p "Args: " file_1 num_sol_1 file_2 num_sol_2 skip_file_2

if [ ! -f "$curr_dir/.$file_1" ]; then
    echo "$file_1 not found"
    exit 1
elif [ ! -f "$curr_dir/.$file_2" ]; then
    echo "$file_2 not found"
    exit 1
else
    echo "Cleaning up solutions..."
fi

cd "$curr_dir/../bin"

if [ ! -d "$curr_dir/../output" ]; then
    mkdir "$curr_dir/../output"
fi

./clean_sol "$curr_dir/.$file_1" $num_sol_1 "$curr_dir/../output/cleaned_1.txt"
if [ $skip_file_2 -eq "0" ]; then
    ./clean_sol $"curr_dir/.$file_2" $num_sol_2 "$curr_dir/../output/cleaned_2.txt" 
fi
echo "Cleaning finished"
echo

echo "Now checking the validity of each solution..."
./check_conditions "$curr_dir/.$file_1" $num_sol_1 1
echo "file 1"
if [ $skip_file_2 -eq "0" ]; then
    ./check_conditions "$curr_dir/.$file_2" $num_sol_2 1
    echo "file 2"
fi
echo "Check finished"
echo

echo "Now checking for duplicates"
./check_duplicates "$curr_dir/.$file_1" $num_sol_1 "$curr_dir/.$file_2" $num_sol_2          \
                   "$curr_dir/../output/duplicates.txt" "$curr_dir/../output/uniques.txt"
echo "Check finished"