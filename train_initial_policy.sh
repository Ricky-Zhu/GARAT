# train on the source env and evaluate on target env
#python train_initial_policy.py --env_name "Hopper-v2" --modified_env_name "HopperMassModified-v2" --args_env_name "Hopper-v2"
#python train_initial_policy.py --env_name "Hopper-v2" --modified_env_name "HopperFrictionModified-v2" --args_env_name "Hopper-v2"
python train_initial_policy.py --env_name "HalfCheetah-v2" --modified_env_name "HalfCheetahModified-v2" --args_env_name "HalfCheetah-v2"
python train_initial_policy.py --env_name "Walker2d-v2" --modified_env_name "Walker2dMassModified-v2" --args_env_name "Walker2d-v2"

# train on the target env to see the optimal performance for a given algo
#python train_initial_policy.py --env_name "HopperMassModified-v2" --modified_env_name "Hopper-v2" --args_env_name "Hopper-v2"
#python train_initial_policy.py --env_name "HopperFrictionModified-v2" --modified_env_name "Hopper-v2" --args_env_name "Hopper-v2"
python train_initial_policy.py --env_name "HalfCheetahModified-v2" --modified_env_name "HalfCheetah-v2" --args_env_name "HalfCheetah-v2"
python train_initial_policy.py --env_name "Walker2dMassModified-v2" --modified_env_name "Walker2d-v2" --args_env_name "Walker2d-v2"