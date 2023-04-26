sim_env="HalfCheetah-v2"
real_env="HalfCheetahModified-v2"

python test.py \
  --target_policy_algo "TRPO" \
  --action_tf_policy_algo "PPO2" \
  --load_policy_path "data/models/TRPO_initial_policy_steps_"$sim_env"_2000000_.pkl" \
  --alpha 1.0 \
  --n_trainsteps_target_policy 100000 \
  --num_cores 1 \
  --sim_env $sim_env \
  --real_env $real_env \
  --n_frames 1 \
  --expt_number 1 \
  --n_grounding_steps 5 \
  --discriminator_epochs 1 \
  --generator_epochs 50 \
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
  --single_batch_size 512 \
  --namespace "CODE_SUBMIT_" \
  --deterministic 0 \
  --n_iters_atp 50 &
  wait
  echo " ~~~ Experiment Completed :) ~~"