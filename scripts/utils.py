import numpy as np
import gym, ast

def parse_args(args):
    args_dict = {}
    for i in range(0, len(args)-1):
        if args[i][:2]=='--' and (args[i+1][:2]!='--'):
            arg_name = args[i].replace('--', '')
            args_dict[arg_name] = args[i+1]
    return args_dict

def rollout_policy_in_envs(real_env,
                           grnd_env,
                           sim_env,
                           target_policy,
                           num_trajs: int,
                           reset_at_every_state = False,
                           hard_init=True,
                           ):

    # a simple function to collect rollouts from the different environments and
    # returns the state trajectories as a dict
    traj_s, traj_r, traj_g = [], [], []

    for traj_count in range(num_trajs):

        # reset environments to same initial state
        real_env.reset()

        if hard_init:
            hard_init_state = np.array([-0.00974552, 0.02286343, -0.48576237, 1.13083801])
            real_env.set_state(hard_init_state[:2], hard_init_state[-2:])
            obs_r = hard_init_state
        else:
            real_env_s0 = real_env.sim.get_state()
            real_env_s0 = np.concatenate((real_env_s0.qpos, real_env_s0.qvel))
            obs_r = real_env_s0

        # grnd_env.reset_state(np.array(obs_r))
        grnd_env.reset()
        grnd_env.set_state(np.array(obs_r)[:2], np.array(obs_r)[-2:])
        obs_g = obs_r
        # obs_g = grnd_env.reset()
        sim_env.reset()
        sim_env.set_state(obs_r[:2], obs_r[-2:])
        obs_s = obs_g
        # obs_s = sim_env.reset()

        done_all, done_g, done_r, done_s = False, False, False, False
        states_r, states_g, states_s = [obs_r], [obs_g], [obs_s]
        # actions_r, actions_g, actions_s = [], [], []

        time_step = 0

        while not done_all:  # collect one trajectory from each environment
            if not done_r:
                action_r, _ = target_policy.predict(obs_r, deterministic=True)
                # actions_r.append(action_r)
                obs_r, rew_r, done_r, _ = real_env.step(action_r)
                states_r.append(obs_r)

            if not done_g:
                if reset_at_every_state:
                    grnd_env.reset()
                    grnd_env.set_state(np.array(obs_r)[:2], np.array(obs_r)[-2:])
                    obs_g = obs_r
                    action_g = action_r
                else:
                    action_g, _ = target_policy.predict(obs_g, deterministic=True)
                # actions_g.append(action_g)
                obs_g, rew_g, done_g, _ = grnd_env.step(action_g)
                states_g.append(obs_g)

            if not done_s:
                if reset_at_every_state:
                    sim_env.set_state(np.array(obs_r)[:2], np.array(obs_r)[-2:])
                    obs_s = obs_r
                    action_s = action_r
                else:
                    action_s, _ = target_policy.predict(obs_s, deterministic=True)
                # actions_s.append(action_s)
                obs_s, rew_s, done_s, _ = sim_env.step(action_s)
                states_s.append(obs_s)


            if done_r is True or done_g is True or done_s is True:
                done_all = True

            # advance time
            time_step += 1

        states_g, states_r, states_s = np.asarray(states_g), np.asarray(states_r), np.asarray(states_s)
        # end of trajectories. Now dump collected trajectories from each environment into traj list
        traj_g.append(states_g)
        traj_s.append(states_s)
        traj_r.append(states_r)

    return np.asarray(traj_r), \
           np.asarray(traj_g), \
           np.asarray(traj_s)


class MinitaurNormalized(gym.ObservationWrapper):
    def __init__(self, env):
        super(MinitaurNormalized, self).__init__(env)
        self.obs_range = abs(self.observation_space.low)
        self.observation_space = gym.spaces.Box(low=-np.ones_like(env.observation_space.low),
                                                high=np.ones_like(env.observation_space.high),
                                                dtype=np.float32)
        self.action_space = env.action_space

    def observation(self, observation):
        return observation/self.obs_range

def read_output(expt_path, file='output.txt'):
    try:
        val = open(expt_path + '/'+file).read().split('\n')[:-1]
        val = [ast.literal_eval(data) for data in val]
        val = np.asarray(val)
        return val
    except:
        return -1

def get_parent_env(env_name):
    parent_env_dict = {
        'Hopper-v2': 'Hopper-v2',
        'HopperModified-v2': 'Hopper-v2',
        'HopperArmatureModified-v2': 'Hopper-v2',
        'DartHopper-v1': 'Hopper-v2',
        'InvertedPendulumModified-v2_old': 'InvertedPendulum-v2',
        'InvertedPendulum-v2': 'InvertedPendulum-v2',
        'Walker2dModified-v2': 'Walker2d-v2',
        'Walker2dFrictionModified-v2': 'Walker2d-v2',
        'Walker2d-v2': 'Walker2d-v2',
        'DartWalker2d-v1': 'Walker2d-v2',
        'MinitaurRealBulletEnv-v0': 'MinitaurBulletEnv-v0',
        'MinitaurRealOnRackBulletEnv-v0': 'MinitaurBulletEnv-v0',
        'MinitaurRealBulletEnvRender-v0': 'MinitaurBulletEnv-v0',
        'MinitaurInaccurateMotorBulletEnv-v0': 'MinitaurBulletEnv-v0',
        'MinitaurInaccurateMotorOnRackBulletEnv-v0': 'MinitaurBulletEnv-v0',
        'MinitaurInaccurateMotorBulletEnvRender-v0': 'MinitaurBulletEnv-v0',
        # 'Ant-v2': 'Ant-v2',
        # 'AntLowGravity-v2': 'Ant-v2',
        'HalfCheetah-v2': 'HalfCheetah-v2',
        'DartHalfCheetah-v1': 'HalfCheetah-v2',
        'AntPyBulletEnv-v0': 'AntPyBulletEnv-v0',
        'AntModifiedBulletEnv-v0': 'AntPyBulletEnv-v0',
    }

    if env_name not in parent_env_dict.keys():
        raise ValueError('The environment has not been added to the mapping yet. Please check scripts/utils/get_parent_env.py')

    return parent_env_dict[env_name]


class MujocoNormalized(gym.ObservationWrapper):
    def __init__(self, env):
        super(MujocoNormalized, self).__init__(env)
        # read this for each environment somehow
        env_name = env.spec.id
        self.max_obs = self._get_max_obs(env_name)

    def observation(self, observation):
        return observation/self.max_obs

    def _get_max_obs(self, env_name):
        # get the parent environment here
        parent_env = get_parent_env(env_name)

        max_obs_dict = {
            'InvertedPendulum-v2': np.array([0.909, 0.098, 1.078, 1.681]),
            'Hopper-v2': np.array([1.587, 0.095, 0.799, 0.191, 0.904,
                                   3.149, 2.767, 2.912, 4.063, 2.581, 10.]),
            # 'Walker2d-v2': np.array([1.547, 0.783, 0.601,0.177,1.322,0.802,0.695,1.182,4.671,3.681,
            #                            5.796,10.,10.,10.,10.,10.,10.]), # old
            'Walker2d-v2': np.array([ 1.35,0.739,1.345,1.605,1.387,1.063,1.202 ,1.339  ,4.988  ,2.863,
                                        10.,10.,10.,10.,10.,10.,10.   ]),
            'MinitaurBulletEnv-v0': np.array([  3.1515927,   3.1515927,   3.1515927,   3.1515927,   3.1515927,
                                                 3.1515927,   3.1515927,   3.1515927, 167.72488  , 167.72488  ,
                                               167.72488  , 167.72488  , 167.72488  , 167.72488  , 167.72488  ,
                                               167.72488  ,   5.71     ,   5.71     ,   5.71     ,   5.71     ,
                                                 5.71     ,   5.71     ,   5.71     ,   5.71     ,   1.01     ,
                                                 1.01     ,   1.01     ,   1.01     ]),
            'AntPyBulletEnv-v0': np.array([0.2100378,  0.5571242,  1. ,        1.0959914,  0.663276,   0.5758094,
                                         0.1813731,  0.2803405,  1.0526485,  1.5340704  ,1.7009641,  1.6335357,
                                         1.1145028 , 1.9834042 , 1.6994406 , 0.8969864 , 1.1013167 , 1.9742222,
                                         1.9255029  ,0.83447146 ,1.0699006  ,1.5556577,  1.8345532  ,1.1241446,
                                         1.         ,1. ,        1.  ,       1.        ]),
            'HalfCheetah-v2': np.array([[ 0.593, 3.618, 1.062, 0.844, 0.837, 1.088, 0.88 , 0.587, 4.165, 3.58,
                                           7.851, 20.837, 25.298, 25.11 , 30.665, 31.541, 15.526]]),

            # 'Humanoid-v2': np.array([]),
        }

        if parent_env not in max_obs_dict.keys():
            raise ValueError('Observation normalization not supported for this environment yet! .')

        return max_obs_dict[parent_env]



def create_env(env_name, normalized, Training=False):
    env = gym.make(env_name)

    if normalized:
        from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load('data/models/env_stats/'+env_name+'.pkl',
                            venv=vec_env)
        vec_env.training = Training
        vec_env.reward_range = env.reward_range

    return env






