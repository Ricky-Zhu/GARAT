import rl_gat.envs
from gym.envs.registration import register
from rl_gat import *
from pybullet_envs import *



register(
    id='HopperMassModified-v2',
    entry_point='rl_gat.envs:HopperMassModifiedEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,

    )

register(
    id='HopperFrictionModified-v2',
    entry_point='rl_gat.envs:HopperFrictionModifiedEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,

)

register(
    id='HalfCheetahModified-v2',
    entry_point='rl_gat.envs:HalfCheetahModifiedEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='Walker2dMassModified-v2',
    entry_point='rl_gat.envs:Walker2dMassModifiedEnv',
    max_episode_steps=1000,
)