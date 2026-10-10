#!/bin/bash

# This script runs main.py 5 separate times with seeds 0 1 2 3 4
# For each set of args

set -e
curr_dir="$(cd "$(dirname "$0")" && pwd)"

identifiers=("optuna" "baseline")
num_iterations=5000
num_epochs=5
num_transitions=200
minibatch_size=20

lr=(6.331696468307533e-05 1e-4)
lr_gamma=(0.14049047497642966 0.32460358795404193)

pol_clip_epsilon=(0.30085053416429375 0.2)
val_clip_epsilon=(0.19085597785347683 0.2)

lr_upd_freq=5     # Should really be called period but I'm no physicist so it doesn't matter ;) 
gae_gamma=(0.930925532933216 0.99)
lmbda=(0.2705703360686772 0.9)
entropy_coef=(0.3222255586355983 0.02)

max_charge=5
max_steps=25

rwd=("abs_err_rwd" "abs_err_rwd")

start=$(date +%s)
for i in {0..1}; do
    echo "Currently on ${identifiers[$i]}"
    echo

    for j in {0..4}; do
        echo "Running trial $((j+1))"
        python3 "$curr_dir/..//main.py" "$num_iterations" "$num_epochs" "$num_transitions" "$minibatch_size"    \
                "${lr[$i]}" "${lr_gamma[$i]}" "${pol_clip_epsilon[$i]}" "${val_clip_epsilon[$i]}"               \
                "5" "${gae_gamma[$i]}" "${lmbda[$i]}" "${entropy_coef[$i]}" "$max_charge" "$max_steps"          \
                "$rwd" "--seed" "$j" "--log" "$curr_dir/../found_charges/${identifiers[$i]}_$j.txt"             \
                "--fig" "$curr_dir/../figures/${identifiers[$i]}_$j.png"
        echo "Finished trial $((j+1))"
        echo
    done
done
end=$(date +%s)

echo "Time elapsed: $(((end - start)))"
