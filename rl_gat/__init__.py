import rl_gat.envs
from gym.envs.registration import register
from rl_gat import *
from pybullet_envs import *
# from pybulletgym import *
#
# register(
#     id='AntModifiedBulletEnv-v0',
#     entry_point='rl_gat.envs:AntBulletModifiedEnv',
#     max_episode_steps=1000,
#     reward_threshold=2500.0,
# )
#
# register(
#     id='AntExtendedLegsBulletEnv-v0',
#     entry_point='rl_gat.envs:AntExtendedLegsBulletEnv',
#     max_episode_steps=1000,
#     reward_threshold=2500.0,
# )
#
# register(
#     id='MinitaurRealBulletEnv-v0',
#     entry_point='pybullet_envs.bullet:MinitaurBulletEnv',
#     max_episode_steps=1000,
#     reward_threshold=15.0,
#     kwargs={
#         'accurate_motor_model_enabled' : True,
#         'env_randomizer' : None,
#         # 'torque_control_enabled' : True,
#     }
# )
#
# register(
#     id='MinitaurRealOnRackBulletEnv-v0',
#     entry_point='pybullet_envs.bullet:MinitaurBulletEnv',
#     max_episode_steps=1000,
#     reward_threshold=15.0,
#     kwargs={
#         'accurate_motor_model_enabled' : True,
#         'env_randomizer' : None,
#         'on_rack' : True,
#         # 'torque_control_enabled' : True,
#     }
# )
#
# register(
#     id='MinitaurRealBulletEnvRender-v0',
#     entry_point='pybullet_envs.bullet:MinitaurBulletEnv',
#     max_episode_steps=1000,
#     reward_threshold=15.0,
#     kwargs={
#         'accurate_motor_model_enabled' : True,
#         'env_randomizer' : None,
#         'render': True ,
# # 'torque_control_enabled': True,
#     }
# )
#
# register(
#     id='MinitaurInaccurateMotorBulletEnv-v0',
#     entry_point='pybullet_envs.bullet:MinitaurBulletEnv',
#     max_episode_steps=1000,
#     reward_threshold=15.0,
#     kwargs={
#         'accurate_motor_model_enabled' : False,
#         'pd_control_enabled' : True,
#         # 'torque_control_enabled': True,
#     }
# )
#
# register(
#     id='MinitaurInaccurateMotorOnRackBulletEnv-v0',
#     entry_point='pybullet_envs.bullet:MinitaurBulletEnv',
#     max_episode_steps=1000,
#     reward_threshold=15.0,
#     kwargs={
#         'accurate_motor_model_enabled' : False,
#         'pd_control_enabled' : True,
#         'on_rack' : True,
#         # 'torque_control_enabled': True,
#     }
# )
#
# register(
#     id='MinitaurInaccurateMotorBulletEnvRender-v0',
#     entry_point='pybullet_envs.bullet:MinitaurBulletEnv',
#     max_episode_steps=1000,
#     reward_threshold=15.0,
#     kwargs={
#         'render' : True,
#         'accurate_motor_model_enabled': False,
#         'pd_control_enabled' : True,
#         # 'torque_control_enabled': True,
#     }
# )
#
#
register(
    id='HopperModified-v2',
    entry_point='rl_gat.envs:HopperModifiedEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
    )
#
# register(
#     id='HopperGravityModified-v2',
#     entry_point='rl_gat.envs:HopperGravityModifiedEnv',
#     max_episode_steps=1000,
#     reward_threshold=3800.0,
#     )
#
# register(
#     id='HopperFrictionModified-v2',
#     entry_point='rl_gat.envs:HopperFrictionModifiedEnv',
#     max_episode_steps=1000,
#     reward_threshold=3800.0,
#     )
#
# register(
#     id='HopperModifiedBulletEnv-v0',
#     entry_point='rl_gat.envs:HopperModifiedBulletEnv',
#     max_episode_steps=1000,
#     reward_threshold=2500.0
#     )
#
# register(
#     id='HopperArmatureModified-v2',
#     entry_point='rl_gat.envs:HopperArmatureModifiedEnv',
#     max_episode_steps=1000,
#     reward_threshold=3800.0,
#     )
#
# register(
#     id='SwimmerModified-v2',
#     entry_point='rl_gat.envs:SwimmerModifiedEnv',
#     max_episode_steps=1000,
#     reward_threshold=360.0,
#     )
#
# register(
#     id='InvertedDoublePendulumModified-v2',
#     entry_point='rl_gat.envs:InvertedDoublePendulumModifiedEnv',
#     max_episode_steps=1000,
#     reward_threshold=9100.0,
#     )
#
# register(
#     id='InvertedPendulumModified-v2',
#     entry_point='rl_gat.envs:InvertedPendulumModifiedEnv',
#     max_episode_steps=1000,
#     reward_threshold=1000.0,
#     )
#
# register(
#     id='InvertedPendulumLong-v2',
#     entry_point='rl_gat.envs:InvertedPendulumLongEnv',
#     max_episode_steps=1000,
#     reward_threshold=1000.0,
#     )
#
# register(
#     id='InvertedPendulumShort-v2',
#     entry_point='rl_gat.envs:InvertedPendulumShortEnv',
#     max_episode_steps=1000,
#     reward_threshold=1000.0,
#     )
#
# register(
#     id='InvertedPendulumLight-v2',
#     entry_point='rl_gat.envs:InvertedPendulumLightEnv',
#     max_episode_steps=1000,
#     reward_threshold=1000.0,
#     )
#
# register(
#     id='InvertedPendulumLightHeavy-v2',
#     entry_point='rl_gat.envs:InvertedPendulumLightHeavyEnv',
#     max_episode_steps=1000,
#     reward_threshold=1000.0,
#     )
#
# register(
#     id='HalfCheetahModified-v2',
#     entry_point='rl_gat.envs:HalfCheetahModifiedEnv',
#     max_episode_steps=1000,
#     reward_threshold=4800.0,
# )
#
# register(
#     id='Walker2dModified-v2',
#     entry_point='rl_gat.envs:Walker2dModifiedEnv',
#     max_episode_steps=1000,
# )
#
# register(
#     id='Walker2dFrictionModified-v2',
#     entry_point='rl_gat.envs:Walker2dFrictionModifiedEnv',
#     max_episode_steps=1000,
# )
#
# register(
#     id='ReacherMassModified-v2',
#     entry_point='rl_gat.envs:ReacherMassModifiedEnv',
#     max_episode_steps=50,
#     reward_threshold=-3.75,
# )
#
# register(
#     id='ReacherDampingModified-v2',
#     entry_point='rl_gat.envs:ReacherDampingModifiedEnv',
#     max_episode_steps=50,
#     reward_threshold=-3.75,
# )
#
# register(
#     id='AntLowGravity-v2',
#     entry_point='rl_gat.envs:AntLowGravityEnv',
#     max_episode_steps=1000,
#     reward_threshold=6000.0,
# )
#
# register(
#     id='AntAmputed-v2',
#     entry_point='rl_gat.envs:AntAmputedEnv',
#     max_episode_steps=1000,
#     reward_threshold=6000.0,
# )
#
# register(
#     id='AntExtended-v2',
#     entry_point='rl_gat.envs:AntExtendedEnv',
#     max_episode_steps=1000,
#     reward_threshold=6000.0,
# )
#
# register(
# 	id='HumanoidModifiedBulletEnv-v0',
# 	entry_point='rl_gat.envs:HumanoidModifiedBulletEnv',
# 	max_episode_steps=1000,
# 	)
#
