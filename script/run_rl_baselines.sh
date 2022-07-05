#!/bin/bash
SRC=./psgi/run_rl_loop.py
# run
default="--seed=1 --num_envs=1 --exp_id 1"
MINING="--algorithm=random --env_id=cooking --num_tasks=100 --graph_param=eval"
INTERVALS="--num_test_episodes 2"
PARAMS="$default $MINING $INTERVALS"
argument="$@"
echo "argument = $argument"

python $SRC $PARAMS $argument
#python -m ipdb -c continue $SRC $PARAMS $argument
