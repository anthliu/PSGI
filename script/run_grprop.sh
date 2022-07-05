#!/bin/bash
SRC=./psgi/run_rl_loop.py
# run
default="--seed=1 --num_envs=16"
MINING="--env_id=mining --num_tasks=440 --graph_param=eval"
GRPROP="--algorithm=grprop --grprop_temp=200 --w_a=3 --beta_a=8 --ep_or=0.8 --temp_or=2.0"
INTERVALS="--num_test_episodes 1"
PARAMS="$default $GRPROP $MINING $INTERVALS"
argument="$@"
echo "argument = $argument"

python $SRC $PARAMS $argument
#python -m ipdb -c continue $SRC $PARAMS $argument
