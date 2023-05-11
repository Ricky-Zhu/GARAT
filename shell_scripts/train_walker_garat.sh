sim_env="Walker2d-v2"
real_env="Walker2dMassModified-v2"

python ../trainer.py \
  --target_policy_algo "TRPO" \
  --action_tf_policy_algo "PPO2" \
  --load_policy_path "../data/models/TRPO_initial_policy_steps_"$sim_env"_2500000_.pkl" \
  --n_trainsteps_target_policy 1000000 \
  --num_cores 1 \
  --sim_env $sim_env \
  --real_env $real_env \
  --n_frames 1 \
  --expt_number 1 \
  --n_grounding_steps 5 \
  --discriminator_epochs 1 \
  --generator_epochs 1 \
  --real_trans 5000 \
  --gsim_trans 5000 \
  --ent_coeff 0.01 \
  --max_kl 3e-4 \
  --clip_range 0.1 \
  --loss_function "GAIL" \
  --eval \
  --disc_lr 3e-3 \
  --atp_lr 3e-4 \
  --nminibatches 2 \
  --noptepochs 10 \
  --compute_grad_penalty \
  --namespace "garat_walker" \
  --deterministic 1 \
  --n_iters_atp 50 &

