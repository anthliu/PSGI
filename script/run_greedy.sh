#!/bin/bash
SRC=./psgi/run_meta_loop.py
# run
default="--seed 1 --num_envs 1 --exp_id 1"
MINING="--env_id cooking --num_trials 2 --graph_param eval --num_adapt_steps 2000"
INTERVALS="--num_test_episodes 16 --num_trial_splits 40"
ALG="--algorithm greedy"
PARAMS="$default $MINING $INTERVALS $ALG"
argument="$@"
echo "argument = $argument"

#python $SRC $PARAMS $argument
python -m ipdb -c continue $SRC $PARAMS $argument
