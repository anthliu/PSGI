#!/bin/bash
SRC=./psgi/run_meta_loop.py
# run
default="--seed 1 --num_envs 1 --exp_id 1"
COOKING="--env_id=cooking --num_trials=10 --num_adapt_steps=2000"
TRAIN="--graph_param=train --label=meta_train --num_trial_splits=1"
INTERVALS="--num_test_episodes=1"
ALG="--algorithm=psgi --exploration=count --ucb_temp 20.0 --neccessary_first"
PARAMS="$default $COOKING $TRAIN $INTERVALS $ALG"
argument="$@"
echo "argument = $argument"

python $SRC $PARAMS $argument
#python -m ipdb -c continue $SRC $PARAMS $argument
