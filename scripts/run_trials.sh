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

actor_lr=(6.010731706199868e-05 1e-4)
critic_lr=(2.0150622953681204e-07 1e-4)
actor_lr_gamma=(0.11737292818213454 0.11737292818213454)
critic_lr_gamma=(0.17496431209947724 0.17496431209947724)
actor_clip_epsilon=(0.5737968512498365 0.2)
critic_clip_epsilon=(0.7325469386729884 0.2)

lr_upd_freq=5     # Should really be called period but I'm no physicist so it doesn't matter ;) 
gae_gamma=(0.2953513202609301 0.99)     # This should probably be >=0.9
lmbda=(0.5738938980970808 0.9)
entropy_coef=(0.6581407850467246 0.02)

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
                "${actor_lr[$i]}" "${critic_lr[$i]}" "${actor_lr_gamma[$i]}" "${critic_lr_gamma[$i]}"           \
                "${actor_clip_epsilon[$i]}" "${critic_clip_epsilon[$i]}" "5" "${gae_gamma[$i]}" "${lmbda[$i]}"  \
                "${entropy_coef[$i]}" "$max_charge" "$max_steps"                                                \
                "$rwd" "--seed" "$j" "--log" "$curr_dir/../found_charges/${identifiers[$i]}_$j.txt"             \
                "--fig" "$curr_dir/../figures/${identifiers[$i]}_$j.png"
        echo "Finished trial $((j+1))"
        echo
    done
done
end=$(date +%s)

echo "Time elapsed: $(((end - start)))"