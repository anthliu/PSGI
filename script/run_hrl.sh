#!/bin/bash
SRC=./psgi/run_meta_loop.py
# run
default="--seed 1 --num_envs 1 --exp_id 1"
MINING="--env_id cooking --num_trials 10 --graph_param eval --num_adapt_steps 2000"
INTERVALS="--num_test_episodes 16 --num_trial_splits 20" # large #episodes since hrl does not use eval actor; but stochastic policy.
ALG="--algorithm hrl --n_step_horizon=4"
PARAMS="$default $MINING $INTERVALS $ALG"
argument="$@"
echo "argument = $argument"

#python $SRC $PARAMS $argument
python -m ipdb -c continue $SRC $PARAMS $argument
