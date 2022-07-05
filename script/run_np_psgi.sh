#!/bin/bash
SRC=./psgi/run_meta_loop.py
# run
default="--seed 1 --num_envs 1 --exp_id 1"
COOKING="--env_id=cooking --num_trials=100 --num_adapt_steps=2000"
EVAL="--graph_param=eval --label=meta_eval --num_trial_splits=20"
INTERVALS="--num_test_episodes=2"
ALG="--algorithm=np_psgi --exploration=count --ucb_temp 20.0 --grprop_temp=200.0 --grprop_sigma=-1.0 --neccessary_first"
PARAMS="$default $COOKING $EVAL $INTERVALS $ALG"
argument="$@"
echo "argument = $argument"

python $SRC $PARAMS $argument
#python -m ipdb -c continue $SRC $PARAMS $argument
