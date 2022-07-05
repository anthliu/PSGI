#!/bin/bash
SRC=./psgi/run_meta_loop.py
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda
# run meta-train
default="--algorithm rlrl --seed 1 --num_envs 36 --label meta_train"
ENV="--env_id playground --num_trials 8000 --graph_param D1_train"
META_PARAMS="--num_adapt_steps 800 --num_test_episodes 1 --num_trial_splits 1"
ALG_PARAMS="--n_step_horizon 10 --learning_rate 0.001 --gae_lambda 0.99 --baseline_cost 0.5 --entropy_cost 1.0 --max_gradient_norm 1.0"
PARAMS="$default $ENV $META_PARAMS $ALG_PARAMS"
argument="$@"
echo "argument = $argument"

#python $SRC $PARAMS $argument
python -m ipdb -c continue $SRC $PARAMS $argument
