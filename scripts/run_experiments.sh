#!/bin/bash

# set which GPU to use
export CUDA_VISIBLE_DEVICES=0

pwd

sim_env="InvertedPendulum-v2"
real_env="InvertedPendulumModified-v2"
#sim_env="Hopper-v2"
#real_env="HopperModified-v2"

#sim_env="HopperArmatureModified-v2"
#real_env="DartHopper-v1"
#real_env="HopperGravityModified-v2"

# script to run experiments
for ((i=1;i<2;i++))
do
  python3.6 test.py \
  --target_policy_algo "TRPO" \
  --action_tf_policy_algo "PPO2" \
  --load_policy_path "data/models/TRPO_initial_policy_steps_"$sim_env"_1000000_.pkl" \
  --alpha 1.0 \
  --n_trainsteps_target_policy 100000 \
  --num_cores 1 \
  --sim_env $sim_env \
  --real_env $real_env \
  --n_frames 1 \
  --expt_number 1 \
  --n_grounding_steps 1 \
  --discriminator_epochs 1 \
  --generator_epochs 1 \
  --real_trajs 1000 \
  --sim_trajs 1000 \
  --real_trans 1024 \
  --gsim_trans 1024 \
  --ent_coeff 0.01 \
  --max_kl 3e-4 \
  --clip_range 0.1 \
  --loss_function "GAIL" \
  --eval \
  --disc_lr 3e-3 \
  --atp_lr 3e-4 \
  --nminibatches 2 \
  --noptepochs 1 \
  --compute_grad_penalty \
  --single_batch_test \
  --single_batch_size 512 \
  --namespace "CODE_SUBMIT_" \
  --deterministic 0 \
  --plot \
  --n_iters_atp 50 &
  wait
  echo " ~~~ Experiment Completed :) ~~"

done

#--compute_grad_penalty \
