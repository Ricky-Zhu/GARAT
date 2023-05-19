# file usage

``state_visitation_collect.py``: collect the points from the state distribution of the import policy.

``trainer.py``: the main doc to run the experiment.

``visualization.py``: t-SNE visual the state distribution of import policies.

``train_initial_policy.py``: train the initial target policy in the source env

# train initial TRPO policy

`python train_initial_policy.py --env_name HalfCheetah-v2 --args_env_name HalfCheetah-v2 --modified_env_name HalfCheetahModified-v2 --time_steps 1000000`