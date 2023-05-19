"""Script good initial policy on some environement"""
import gym

# Filter tensorflow version warnings
import os

# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings

# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging

tf.get_logger().setLevel(logging.ERROR)

from stable_baselines.common.policies import MlpPolicy as mlp_standard
from stable_baselines.sac.policies import FeedForwardPolicy as ffp_sac
from stable_baselines.td3.policies import FeedForwardPolicy as ffp_td3
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines import SAC, TD3, TRPO, PPO2, ACKTR
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common.callbacks import EvalCallback
import numpy as np
import yaml, shutil
import sys
import random

sys.path.append('rl_gat')
from rl_gat.envs import *

ALGO = TRPO
# set the environment here :
ENV_NAME = 'Hopper-v2'
MODIFIED_ENV_NAME = 'HopperModified-v2'
TIME_STEPS = 2500000
NOISE_VALUE = 0.0
SAVE_BEST_FOR_20 = False
MUJOCO_NORMALIZE = False
NORMALIZE = False

# create model save folder
if not os.path.exists('scripts/data/models/'):
    os.makedirs('scripts/data/models/')


# define the model name for creating folders and saving the models
def create_model_name(env_name,
                      time_steps):
    if SAVE_BEST_FOR_20:
        model_name = "data/models/" + ALGO.__name__ + "_initial_policy_steps_" + env_name + "_" + str(
            time_steps) + "_best_.pkl"
    else:
        model_name = "data/models/" + ALGO.__name__ + "_initial_policy_steps_" + env_name + "_" + str(
            time_steps) + "_.pkl"
    return model_name


def evaluate_policy_on_env(env,
                           model,
                           render=True,
                           iters=1,
                           deterministic=True,
                           save_the_optim_traj_states=False
                           ):
    # model.set_env(env)
    return_list = []
    state_visit = []
    for i in range(iters):
        return_val = 0
        done = False
        obs = env.reset()
        while not done:
            if save_the_optim_traj_states:
                state_visit.append(obs)
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, rewards, done, info = env.step(action)
            return_val += rewards
            if render:
                env.render()
                # time.sleep(0.01)

        if not i % 15: print('Iteration ', i, ' done.')
        return_list.append(return_val)
    # sample 2000 data points to represent state visitation
    if save_the_optim_traj_states:
        state_visit_to_save = random.sample(state_visit, k=2000)
    print('***** STATS FOR THIS RUN *****')
    print('MEAN : ', np.mean(return_list))
    print('STD : ', np.std(return_list))
    print('******************************')
    if save_the_optim_traj_states:
        return np.mean(return_list), np.std(return_list) / np.sqrt(len(return_list)), state_visit_to_save
    else:
        return np.mean(return_list), np.std(return_list) / np.sqrt(len(return_list))


def train_initial_policy(
        model_name,
        algo=ALGO,
        env_name=ENV_NAME,
        modified_env_name=MODIFIED_ENV_NAME,
        args_env_name=ENV_NAME,
        time_steps=TIME_STEPS):
    """Uses the specified algorithm on the target environment"""
    print("Using algorithm : ", algo.__name__)
    print("Model saved as : ", "data/models/" + algo.__name__ + "_initial_policy_" + env_name + "_.pkl")

    # define the environment here
    env = gym.make(env_name)

    env = DummyVecEnv([lambda: env])

    # loading the args for different envs
    with open('data/target_policy_params.yaml') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    args = args[algo.__name__][args_env_name]

    if algo.__name__ == "SAC":

        class CustomPolicy(ffp_sac):
            def __init__(self, *args, **kwargs):
                super(CustomPolicy, self).__init__(*args, **kwargs,
                                                   feature_extraction="mlp", layers=[256, 256])

        model = SAC(CustomPolicy, env,
                    verbose=1,
                    tensorboard_log='data/TBlogs/initial_policy_training',
                    batch_size=args['batch_size'],
                    buffer_size=args['buffer_size'],
                    ent_coef=args['ent_coef'],
                    learning_starts=args['learning_starts'],
                    learning_rate=args['learning_rate'],
                    train_freq=args['train_freq'],
                    )
    elif algo.__name__ == "TD3":
        print('Initializing TD3 with RLBaselinesZoo hyperparameters .. ')
        # hyperparameters suggestions from :
        # https://github.com/araffin/rl-baselines-zoo/blob/master/trained_agents/td3/HopperBulletEnv-v0/config.yml
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=float(args['noise_std']) * np.ones(n_actions))

        class CustomPolicy2(ffp_td3):
            def __init__(self, *args, **kwargs):
                super(CustomPolicy2, self).__init__(*args, **kwargs,
                                                    feature_extraction="mlp", layers=[400, 300])

        model = TD3(CustomPolicy2, env,
                    verbose=1,
                    tensorboard_log='data/TBlogs/initial_policy_training',
                    batch_size=args['batch_size'],
                    buffer_size=args['buffer_size'],
                    gamma=args['gamma'],
                    gradient_steps=args['gradient_steps'],
                    learning_rate=args['learning_rate'],
                    learning_starts=args['learning_starts'],
                    action_noise=action_noise,
                    train_freq=args['train_freq'],
                    )

    elif algo.__name__ == "TRPO":

        # hyperparameters suggestions from :
        # https://github.com/araffin/rl-baselines-zoo/blob/master/trained_agents/sac/HopperBulletEnv-v0/config.yml
        model = TRPO(mlp_standard, env,
                     verbose=0,
                     tensorboard_log='data/TBlogs/initial_policy_training',
                     timesteps_per_batch=args['timesteps_per_batch'],
                     lam=args['lam'],
                     max_kl=args['max_kl'],
                     gamma=args['gamma'],
                     vf_iters=args['vf_iters'],
                     vf_stepsize=args['vf_stepsize'],
                     entcoeff=args['entcoeff'],
                     cg_damping=args['cg_damping'],
                     cg_iters=args['cg_iters']
                     )

    elif algo.__name__ == "ACKTR":
        print('Initializing ACKTR')
        model = ACKTR(mlp_standard,
                      env,
                      verbose=1,
                      n_steps=128,
                      ent_coef=0.01,
                      lr_schedule='constant',
                      learning_rate=0.0217,
                      max_grad_norm=0.5,
                      gamma=0.99,
                      vf_coef=0.946)

    elif algo.__name__ == "PPO2":
        print('Initializing PPO2')
        print('Num envs : ', env.num_envs)
        model = PPO2(mlp_standard,
                     env,
                     n_steps=int(args['n_steps'] / env.num_envs),
                     nminibatches=args['nminibatches'],
                     lam=args['lam'],
                     gamma=args['gamma'],
                     ent_coef=args['ent_coef'],
                     noptepochs=args['noptepochs'],
                     learning_rate=args['learning_rate'],
                     cliprange=args['cliprange'],
                     verbose=1,
                     tensorboard_log='data/TBlogs/initial_policy_training',
                     )

    else:
        pass
    # change model name if using normalization
    if NORMALIZE:
        model_name = model_name.replace('.pkl', 'normalized_.pkl')

    elif MUJOCO_NORMALIZE:
        model_name = model_name.replace('.pkl', 'mujoco_norm_.pkl')

    # if SAVE_BEST_FOR_20:
    #     model.learn(total_timesteps=time_steps,
    #                 tb_log_name=model_name,
    #                 log_interval=10,
    #                 callback=eval_callback)
    #     save_the_model()
    #     model_name = model_name.replace('best_', '')
    #     model.save(model_name)

    else:
        model.learn(total_timesteps=time_steps,
                    tb_log_name=model_name.split('/')[-1],
                    log_interval=10, )
        model.save(model_name)
        print('--------------' + env_name + '----------------')
        optim_state_visit = evaluate_policy_on_env(env, model, render=False, iters=10, save_the_optim_traj_states=True)[
            -1]
        # save the optim state visit
        optim_state_visit = np.asarray(optim_state_visit)
        np.save('data/optim_state_visit_{}_{}.npy'.format(env_name, TIME_STEPS), optim_state_visit)
        # then evaluate the trained policy in the source domain to the modified env to see the performance desprepency
        env_modified = DummyVecEnv([lambda: gym.make(modified_env_name)])
        print('--------------' + modified_env_name + '----------------')
        evaluate_policy_on_env(env_modified, model, render=False, iters=10)
        print('******************************')
        print('\n')

    # save the environment params
    if NORMALIZE:
        # env.save(model_name.replace('.pkl', 'stats_.pkl'))
        env.save('data/models/env_stats/' + env_name + '.pkl')

    print('done :: ', model_name)
    exit()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='train initial policy in the default env and evaluate on also the modified env')
    parser.add_argument('--env_name', default='HalfCheetah-v2', type=str, help='the source env')
    parser.add_argument('--args_env_name', default='HalfCheetah-v2', type=str, help='the env args is selected for')
    parser.add_argument('--modified_env_name', default='HalfCheetahModified-v2', type=str, help='the target env')
    parser.add_argument('--time_steps', default=5000000, type=int, help='total time steps to learn')

    args = parser.parse_args()
    model_name = create_model_name(env_name=args.env_name,
                                   time_steps=args.time_steps)
    train_initial_policy(model_name=model_name,
                         env_name=args.env_name,
                         modified_env_name=args.modified_env_name,
                         args_env_name=args.args_env_name,
                         time_steps=args.time_steps)
