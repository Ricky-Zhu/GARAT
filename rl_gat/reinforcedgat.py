"""Implements Reinforced-GAT algorithm

Implementation of Reinforced-GAT algorithm
for OpenAI Gym environments.
"""

# pylint: disable=invalid-name,too-many-locals, wrong-import-position, too-many-arguments, not-callable
# Suppress these warnings to allow for certain machine learning conventions

import os, shutil
import numpy as np
from stable_baselines import PPO2, SAC, TD3, TRPO
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy as OtherMlpPolicy, FeedForwardPolicy, MlpLstmPolicy
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.buffers import ReplayBuffer
import gym
from termcolor import cprint
import random
from gym import spaces
import torch

torch.backends.cudnn.deterministic = True
import torch.nn as nn
from tqdm import trange
import matplotlib.pyplot as plt
from rl_gat.gat import collect_gym_trajectories, train_model_es, apply_norm
from rl_gat.gat import Net, unapply_norm
import yaml, sys
from scripts.utils import MujocoNormalized
import pickle


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)

# DEBUG = False

# Number of CPU Cores
NUM_CORES = 8
# Number of trajectories to collect on 'real' environment
# NUM_REAL_WORLD_TRAJECTORIES = 20 if DEBUG else 100
# Number of trajectories to collect on simulated environment
# NUM_SIM_WORLD_TRAJECTORIES = 20 if DEBUG else 100
# Max number of epochs to train models
MAX_EPOCHS = 5
# Max number of epochs to train forward model
MAX_EPOCHS_FWD = 100
# Fraction of dataset to use as validation
VALIDATION_SIZE = 0.2
# No. of timesteps to train target policy
LEARN_TIMESTEPS = 1000000
# Ratio between the max change in action by the action transformer and the
# action range of the actual simulator (smaller values are more restrictive
# but lead to faster learning)
ACTION_TF_RATIO = 2.0

# No. of timesteps to train action transformer policy
MAX_EPOCHS_ATP = 5
# BATCH_SIZE = 10000
# LEARN_ATP_TIMESTEPS = BATCH_SIZE*MAX_EPOCHS_ATP

# No. of threads to used during RL step
NUM_RL_THREADS = 1


class Discriminator(torch.nn.Module):
    """network that defines the Discriminator"""

    def __init__(self, n_feature, n_hidden, action_space, activations=nn.Tanh):
        super(Discriminator, self).__init__()

        self.fc_in = nn.Linear(n_feature, n_hidden)
        self.fc_h1 = nn.Linear(n_hidden, n_hidden)
        self.fc_h2 = nn.Linear(n_hidden, n_hidden)
        # self.fc_h3 = nn.Linear(n_hidden, n_hidden)
        self.fc_out = nn.Linear(n_hidden, 1)

        self.single_fc = nn.Linear(n_feature, 1)

        self.activations = activations
        self.action_space = action_space

        torch.nn.init.xavier_uniform_(self.fc_in.weight)
        torch.nn.init.xavier_uniform_(self.fc_h1.weight)
        torch.nn.init.xavier_uniform_(self.fc_h2.weight)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)
        torch.nn.init.xavier_uniform_(self.single_fc.weight)

    # pylint: disable=arguments-differ
    def forward(self, x):
        out = self.activations()(self.fc_in(x))
        out = self.activations()(self.fc_h1(out))
        out = self.activations()(self.fc_h2(out))
        # out = self.activations()(self.fc_h3(out))
        out = self.fc_out(out)

        skip = nn.ReLU()(self.single_fc(x))
        out = out + skip

        return out


class ATPEnv(gym.Wrapper):
    """
    Defines the Action Transformer Policy's environment
    """

    def __init__(self,
                 env,
                 target_policy,
                 discriminator=None,
                 fwd_model=None,
                 disc_norm=None,
                 fwd_norm=(0, 1),
                 lam=0.0,
                 beta=1.0,
                 atr=ACTION_TF_RATIO,
                 train_noise=0.0,
                 device='cpu',
                 loss='GAIL',
                 normalizer=None,
                 frames=1,
                 data_collection_mode=False,
                 expt_path=None,
                 ):
        super(ATPEnv, self).__init__(env)
        self.target_policy = target_policy
        self.normalizer = normalizer
        if discriminator is None:
            assert fwd_model is not None, \
                "At least one of discriminator or fwd_model must be specified."
            self.beta = 0.0
        elif fwd_model is None:
            self.beta = 1.0
        else:
            self.beta = beta

        self.expt_path = expt_path

        self.device = device
        self.discriminator = discriminator
        self.fwd_model = fwd_model
        self.input_norm = disc_norm
        self.fwd_norm = fwd_norm
        self.lam = lam
        self.loss = loss

        low = np.append(self.env.observation_space.low,
                        self.env.action_space.low)
        high = np.append(self.env.observation_space.high,
                         self.env.action_space.high)
        self.obs_size = low.shape[0]
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.env_max_act = (self.env.action_space.high - self.env.action_space.low) / 2

        max_act = (self.env.action_space.high - self.env.action_space.low) / 2 * atr
        self.action_space = spaces.Box(-max_act, max_act, dtype=np.float32)
        self.train_noise = train_noise
        self.frames = frames
        if data_collection_mode:
            self.data_collection_mode = True
            self.Ts = []
        else:
            self.data_collection_mode = False
        # These are set when reset() is called
        self.latest_obs = None
        self.latest_act = None
        self.prev_frames = None
        self.time_step_counter = 0

    def refresh_disc(self, target_policy, discriminator, disc_norm):
        self.target_policy = target_policy
        self.discriminator = discriminator
        self.input_norm = disc_norm

    def reset(self, **kwargs):
        """Reset function for the wrapped environment"""
        self.latest_obs = self.env.reset(**kwargs)
        if self.normalizer is not None:
            # self.latest_obs = self.normalizer.normalize_obs(self.latest_obs)
            self.latest_obs = self.normalizer.reset(**kwargs)[0]

        self.latest_act, _ = self.target_policy.predict(self.latest_obs, deterministic=True)

        # create empty list and pad with zeros
        self.prev_frames = []
        self.prev_actions = []
        for _ in range(self.frames - 1):
            self.prev_frames.extend(np.zeros_like(np.hstack((self.latest_obs, self.latest_act))))
        self.prev_frames.extend(np.hstack((self.latest_obs, self.latest_act)))

        self.time_step_counter = 0

        if self.data_collection_mode:
            self.T = []
            self.new_traj = True

        # Return the observation for THIS environment
        return np.append(self.latest_obs, self.latest_act)

    def step(self, action):
        """
        Step function for the wrapped environment
        """
        self.time_step_counter += 1

        if self.data_collection_mode: self.T.append((self.latest_obs, self.latest_act))

        # input action is the delta transformed action for this Environment
        transformed_action = action + self.latest_act
        transformed_action = np.clip(transformed_action, -self.env_max_act, self.env_max_act)

        if self.normalizer is None:
            sim_next_state, _, sim_done, info = self.env.step(transformed_action)
        else:
            sim_next_state, _, sim_done, info = self.normalizer.step(transformed_action)
            sim_next_state, sim_done, info = sim_next_state[0], sim_done[0], info[0]

        # get target policy action
        # if self.normalizer is not None:
        #     sim_next_state = self.normalizer.normalize_obs(sim_next_state)
        target_policy_action, _ = self.target_policy.predict(sim_next_state, deterministic=True)

        ###### experimenting with adding noise while training ATPEnv ######
        # target_policy_action = target_policy_action + np.random.normal(0, self.train_noise**0.5, target_policy_action.shape[0])

        concat_sa = np.append(sim_next_state, target_policy_action)

        # use discriminator only if beta is > 0.0
        if self.beta > 0.0:
            # discriminator reward
            concat_sas = np.concatenate((self.prev_frames,  # self.latest_obs,
                                         # self.latest_act,
                                         sim_next_state))

            concat_sas = apply_norm(concat_sas, self.input_norm[0])
            concat_sas = torch.tensor(concat_sas).float().to(self.device)
            disc_rew_logit = self.discriminator(concat_sas)
            disc_rew = torch.nn.Sigmoid()(disc_rew_logit)  # pass through sigmoid
            disc_rew = disc_rew.detach().to(self.device).numpy()

            # log-ify the discriminator value
            # 1e-8 term was used by faraz in the GAIfO implementation
            if self.loss == 'GAIL':
                disc_rew = -np.log(1.0 - disc_rew + 1e-8)[0]

            # non saturating loss function
            elif self.loss == 'NSGAIL':
                disc_rew = np.log(disc_rew + 1e-8)[0]

            # testing least squares GAN [Same result as ^^ - doesnt work :(]
            # disc_rew = -((disc_rew.detach().cpu().numpy()-1)[0]**2)

            # testing AIRL formulation
            elif self.loss == 'AIRL':
                disc_rew = (np.log(disc_rew + 1e-8) - np.log(1 - disc_rew + 1e-8))[0]

            # testing FAIRL formulation
            elif self.loss == 'FAIRL':
                disc_rew = (np.log(disc_rew + 1e-8) - np.log(1 - disc_rew + 1e-8))[0]
                disc_rew = -disc_rew * np.exp(disc_rew)

            elif self.loss == 'WGAN':
                disc_rew_logit = disc_rew_logit.detach().to(self.device).numpy()
                disc_rew = 1.0 * disc_rew_logit

            # penalize if action crosses max_act
            # disc_rew = disc_rew - np.linalg.norm(np.minimum(0, self.max_act-abs(transformed_action)))**2
        else:
            disc_rew = 0.0

        # # run forward model only if beta <1.0
        # if self.beta < 1.0:
        #     # fwd model reward
        #     concat_ps_a = np.concatenate((self.latest_obs,
        #                                   self.latest_act))
        #     concat_ps_a = apply_norm(concat_ps_a,
        #                              self.fwd_norm[0])
        #     concat_ps_a = torch.tensor(concat_ps_a).float().to(self.device)
        #     pred_delta_next_state = self.fwd_model(concat_ps_a)
        #     pred_delta_next_state = pred_delta_next_state.detach().to(self.device).numpy()
        #     pred_delta_next_state = unapply_norm(pred_delta_next_state,
        #                                          self.fwd_norm[1])
        #     pred_next_state = pred_delta_next_state + self.latest_obs
        #     fwd_rew = -1*np.sum((pred_next_state-sim_next_state)**2)
        # else:
        #     fwd_rew = 0.0

        self.latest_obs = sim_next_state
        self.latest_act = target_policy_action

        self.prev_frames = self.prev_frames[self.obs_size:]
        self.prev_frames.extend(np.hstack((self.latest_obs, self.latest_act)))

        if sim_done and self.data_collection_mode:
            self.T.append((self.latest_obs, None))
            self.Ts.append(self.T)
            self.T = []
            self.new_traj = False
            pickle.dump(self.Ts, open(self.expt_path + '/fake_data.p', "wb"))

        # TODO: we should figure out what to do with sim_reward
        output_reward = disc_rew  # + self.lam*sim_reward + (1-self.beta)*fwd_rew #- 0.1*np.sum(transformed_action**2)
        return concat_sa, output_reward, sim_done, info

    def get_fake_trajs(self):
        if self.new_traj: self.Ts.append(self.T)
        return self.Ts

    def reset_trajs(self):
        del self.Ts
        self.Ts = []

    def close(self):
        self.env.close()
        if self.normalizer is not None:
            self.normalizer.close()


class GroundedEnv(gym.ActionWrapper):
    """
    Defines the grounded environment, from the perspective of the target policy
    """

    # pylint: disable=abstract-method
    def __init__(self,
                 env,
                 action_tf_policy,
                 # action_tf_env,
                 alpha=1.0,
                 debug_mode=True,
                 normalizer=None,
                 data_collection_mode=False,
                 use_deterministic=True,
                 atp_policy_noise=0.0,
                 ):
        super(GroundedEnv, self).__init__(env)
        self.debug_mode = debug_mode
        self.action_tf_policy = action_tf_policy
        # self.atp_env = action_tf_env
        # self.action_tf_policy.set_env(self.atp_env)
        self.alpha = alpha
        self.normalizer = normalizer
        if self.debug_mode:
            self.transformed_action_list = []
            self.raw_actions_list = []
        if data_collection_mode:
            self.data_collection_mode = True
            self.Ts = []
        else:
            self.data_collection_mode = False
        # These are set when reset() is called
        self.latest_obs = None
        # self.prev_frames = None
        self.time_step_counter = 0
        self.high = env.action_space.high
        self.low = env.action_space.low
        self.use_deterministic = use_deterministic
        self.atp_policy_noise = atp_policy_noise

    def reset(self, **kwargs):
        if self.normalizer is not None:
            # self.latest_obs = self.normalizer.normalize_obs(self.latest_obs)
            self.latest_obs = self.normalizer.reset(**kwargs)
            self.latest_obs = self.latest_obs[0]
        else:
            self.latest_obs = self.env.reset(**kwargs)

        if self.data_collection_mode:
            self.T = []

        # self.prev_frames = [self.latest_obs for _ in range(NUM_FRAMES_INPUT)]
        self.time_step_counter = 0
        return self.latest_obs

    def reset_state(self, state: np.array):
        assert 'InvertedPendulum' in self.env.unwrapped.spec.id, "Unsupported gym environment"
        assert state.shape[0] == 4, "Array of dimension 4 required"
        self.env.reset()
        self.env.set_state(np.array(state[:2]), np.array(state[-2:]))
        self.latest_obs = state
        # if self.normalizer is not None:
        #     self.latest_obs = self.normalizer.normalize_obs(self.latest_obs)

    def step(self, action):
        self.time_step_counter += 1

        if self.data_collection_mode: self.T.append((self.latest_obs, action))

        # TODO: add more frames here ?
        concat_sa = np.append(self.latest_obs, action)
        # change made : lets assume the output of the ATP is \delta_a_t
        # concat_sa = self.atp_env.normalize_obs(concat_sa)
        delta_transformed_action, _ = self.action_tf_policy.predict(concat_sa, deterministic=self.use_deterministic)

        # NEW : experimenting with adding noise here
        delta_transformed_action += np.random.normal(0, self.atp_policy_noise ** 0.5, delta_transformed_action.shape[0])

        # print('delta : ',delta_transformed_action)

        transformed_action = action + self.alpha * delta_transformed_action
        # transformed_action = action + delta_transformed_action

        transformed_action = np.clip(transformed_action, self.low, self.high)

        # transformed_action = delta_transformed_action
        if self.normalizer is not None:
            self.latest_obs, rew, done, info = self.normalizer.step(transformed_action)
            self.latest_obs = self.latest_obs[0]
            rew, done, info = rew[0], done[0], info[0]
        else:
            self.latest_obs, rew, done, info = self.env.step(transformed_action)

        # if self.normalizer is not None:
        #     self.latest_obs = self.normalizer.normalize_obs(self.latest_obs)
        # self.prev_frames = self.prev_frames[1:]+[self.latest_obs]

        if self.debug_mode and self.time_step_counter <= 1e4:
            self.transformed_action_list.append(transformed_action)
            self.raw_actions_list.append(action)

        # change the reward to be a function of the input action and
        # not the transformed action
        if 'Hopper' in self.env.unwrapped.spec.id:
            rew = rew - 1e-3 * np.square(action).sum() + 1e-3 * np.square(transformed_action).sum()
        elif 'HalfCheetah' in self.env.unwrapped.spec.id:
            rew = rew - 0.1 * np.square(action).sum() + 0.1 * np.square(transformed_action).sum()
        elif 'Swimmer' in self.env.unwrapped.spec.id:
            rew = rew - 0.0001 * np.square(action).sum() + 0.0001 * np.square(transformed_action).sum()
        elif 'Reacher' in self.env.unwrapped.spec.id:
            rew = rew - np.square(action).sum() + np.square(transformed_action).sum()
        elif 'Ant' in self.env.unwrapped.spec.id:
            rew = rew - 0.5 * np.square(action).sum() + 0.5 * np.square(transformed_action).sum()
        elif 'Humanoid' in self.env.unwrapped.spec.id:
            rew = rew - 0.1 * np.square(action).sum() + 0.1 * np.square(transformed_action).sum()
        elif 'Pusher' in self.env.unwrapped.spec.id:
            rew = rew - np.square(action).sum() + np.square(transformed_action).sum()
        elif 'Walker2d' in self.env.unwrapped.spec.id:
            rew = rew - 1e-3 * np.square(action).sum() + 1e-3 * np.square(transformed_action).sum()
        elif 'HumanoidStandup' in self.env.unwrapped.spec.id:
            rew = rew - 0.1 * np.square(action).sum() + 0.1 * np.square(transformed_action).sum()

        if done and self.data_collection_mode:
            self.T.append((self.latest_obs, None))
            self.Ts.extend(self.T)
            self.T = []

        return self.latest_obs, rew, done, info

    def get_trajs(self):
        return self.Ts

    def sim2sim_plot_action_transformation_graph(
            self,
            expt_path,
            grounding_step,
            show_plot=False,
            plot_e3=False, ):
        """Graphs transformed actions vs input actions"""
        num_action_space = self.env.action_space.shape[0]
        action_low = self.env.action_space.low[0]
        action_high = self.env.action_space.high[0]

        self.raw_actions_list = np.asarray(self.raw_actions_list)
        self.transformed_action_list = np.asarray(self.transformed_action_list)
        colors = ['go', 'bo', 'ro', 'mo', 'yo', 'ko']

        if num_action_space > len(colors):
            print("Unsupported Action space shape.")
            return

        # plotting the data points starts here
        fig = plt.figure(figsize=(int(10 * num_action_space), 10))
        for act_num in range(num_action_space):
            ax = fig.add_subplot(1, num_action_space, act_num + 1)
            ax.plot(self.raw_actions_list[:, act_num], self.transformed_action_list[:, act_num], colors[act_num])
            ax.plot([action_low, action_high], [action_low, action_high], 'k-')  # black line
            ax.plot([action_low, action_high], [0, 0], 'r-')  # red lines
            ax.plot([0, 0], [action_low, action_high], 'r-')  # red lines

            ax.title.set_text('Action Space # :' + str(act_num + 1) + '/' + str(num_action_space))
            ax.set(xlabel='Input Actions', ylabel='Transformed Actions', xlim=[action_low, action_high],
                   ylim=[action_low, action_high])
        plt.suptitle('Plot of Input action $a_t$ vs Transformed action $\\tilde{a}_t$')
        if 'data/models/garat' in expt_path:
            plt.savefig(os.getcwd() + '/' + expt_path + '/' + str(grounding_step) + '_e4_.png')
        else:
            plt.savefig(expt_path + '/' + str(grounding_step) + '_e4_.png')
        if show_plot: plt.show()
        plt.close()

        # if not plot_e3:
        #     return
        #
        # fig = plt.figure(figsize=(int(10*num_action_space), 10))
        # for act_num in range(num_action_space):
        #     ax = fig.add_subplot(1, num_action_space, act_num+1)
        #     ax.plot(self.raw_actions_list[:1000, act_num], self.transformed_action_list[:1000, act_num], colors[act_num])
        #     ax.plot([action_low, action_high], [action_low, action_high], 'k-') # black line
        #     ax.title.set_text('Action Space # :'+ str(act_num+1)+'/'+str(num_action_space))
        #     ax.set(xlabel = 'Input Actions', ylabel = 'Transformed Actions', xlim=[-1, 1], ylim=[action_low, action_high])
        # plt.suptitle('Plot of Input action $a_t$ vs Transformed action $\\tilde{a}_t$')
        # plt.savefig(os.getcwd()+'/'+expt_path+'/'+str(grounding_step)+'_e3_.png')
        # plt.close()

    def close(self):
        self.env.close()
        if self.normalizer is not None:
            self.normalizer.close()


class ReinforcedGAT:
    """Implements Reinforced-GAT"""

    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 expt_path,
                 expt_label=None,
                 sim_env_name='Hopper-v2',
                 real_env_name='HopperModified-v2',
                 frames=1,
                 num_cores=NUM_CORES,
                 num_rl_threads=NUM_RL_THREADS,
                 load_policy=None,
                 algo="TRPO",
                 atp_algo="TRPO",
                 debug=False,
                 real_trajs=50,
                 sim_trajs=50,
                 use_cuda=False,
                 real_trans=50000,
                 gsim_trans=50000,
                 tensorboard=False,
                 atp_loss_function='GAIL',
                 single_batch_size=None,
                 ):

        self.single_batch_size = single_batch_size
        self.tp_algo = algo
        self.atp_algo = atp_algo
        # set which device to use
        self.device = 'cuda' if use_cuda else 'cpu'

        if expt_label is None:
            print('An experiment label is missing. ')
            self.expt_label = input('Enter an experiment label : ')
        else:
            self.expt_label = expt_label

        # create folder to save all the plots and the models
        self.expt_path = expt_path

        # Number of trajectories to collect on 'real' environment
        self.NUM_REAL_WORLD_TRAJECTORIES = 20 if debug else real_trajs
        # Number of trajectories to collect on simulated environment
        self.NUM_SIM_WORLD_TRAJECTORIES = 20 if debug else sim_trajs
        # instead use num of transitions from real and sim ;)
        self.real_trans = real_trans
        self.gsim_trans = gsim_trans

        self.env_name = sim_env_name
        self.real_env_name = real_env_name
        self.frames = frames
        self.num_cores = num_cores
        self.num_rl_threads = num_rl_threads

        # using custom mujoco normalization scheme
        # which in the experiment is not used
        self.mujoco_norm = False
        if 'mujoco_norm' in load_policy: self.mujoco_norm = True

        env = gym.make(self.real_env_name)
        if self.mujoco_norm: env = MujocoNormalized(env)
        self.real_env = DummyVecEnv([lambda: env])

        env = gym.make(self.env_name)
        if self.mujoco_norm: env = MujocoNormalized(env)
        self.sim_env = DummyVecEnv([lambda: env])

        # initialize target policy
        self.load_policy = load_policy
        self.algo = algo
        self.saved_env = env
        self._init_target_policy(load_policy, algo, env, tensorboard)

        # define the Grounded Action Transformer models here
        self.grounded_sim_env = None

        # lists to reuse experience from previous grounding steps
        self.data_x_list = []
        self.data_y_list = []

        self.real_X_list, self.real_Y_list = [], []
        self.fwd_X_list, self.fwd_Y_list = [], []

    def _randomize_target_policy(self, algo, env=None):

        cprint('### ~~~ RESETTING TARGET POLICY ~~~ ###', 'red', 'on_blue')

        with open('data/target_policy_params.yaml') as file:
            args = yaml.load(file, Loader=yaml.FullLoader)

        if algo == "PPO2":
            cprint('Using PPO2 as the Target Policy Algo', 'yellow')
            args = args['PPO2'][self.env_name]
            self.target_policy = PPO2(
                OtherMlpPolicy,
                env=DummyVecEnv([lambda: gym.make(self.env_name)]),
                verbose=1,
                n_steps=args['n_steps'],
                nminibatches=args['nminibatches'],
                lam=args['lam'],
                gamma=args['gamma'],
                noptepochs=args['noptepochs'],
                ent_coef=args['ent_coef'],
                learning_rate=args['learning_rate'],
                cliprange=args['cliprange'],
            )

        elif algo == "TRPO":
            cprint('Using TRPO as the Target Policy Algo', 'yellow')
            args = args['TRPO'][self.env_name]
            self.target_policy = TRPO(
                OtherMlpPolicy,
                env=DummyVecEnv([lambda: gym.make(self.env_name)]),
                verbose=1,
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

    def _init_target_policy(self, load_policy, algo, env=None, tensorboard=False):

        if env is None: env = self.saved_env

        if load_policy is None:
            print('LOADING -RANDOM- INITIAL POLICY')
            if algo == "PPO2":
                self.target_policy = PPO2(
                    OtherMlpPolicy,
                    # env=DummyVecEnv([lambda : gym.make(self.env_name)]),
                    verbose=1,
                    # tensorboard_log='data/TBlogs/' + self.env_name,
                )
            elif algo == "TD3":
                n_actions = self.sim_env.action_space.shape[-1]
                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                self.target_policy = TD3(
                    MlpPolicy,
                    env=DummyVecEnv([lambda: gym.make(self.env_name)]),
                    # tensorboard_log='data/TBlogs/' + self.env_name,
                    verbose=1,
                    batch_size=128,
                    gamma=0.99,
                    learning_rate=0.001,
                    action_noise=action_noise,
                    buffer_size=1000000
                )
            else:
                raise NotImplementedError("Algo " + algo + " not supported")
        else:
            print('LOADING -PRETRAINED- INITIAL POLICY')
            with open('data/target_policy_params.yaml') as file:
                args = yaml.load(file, Loader=yaml.FullLoader)

            if 'normalize' in load_policy:
                self._init_normalization_stats()
            else:
                self.target_policy_norm_obs = None

            if algo == "SAC":
                args = args['SAC'][self.env_name]
                self.target_policy = SAC.load(
                    load_policy,
                    # env=DummyVecEnv([lambda: env]),
                    # tensorboard_log='data/TBlogs/'+self.env_name,
                    verbose=1,
                    batch_size=args['batch_size'],
                    buffer_size=args['buffer_size'],
                    ent_coef=args['ent_coef'],
                    learning_rate=args['learning_rate'],
                    learning_starts=args['learning_starts'],
                )

            elif algo == "PPO2":
                args = args['PPO2'][self.env_name]

                self.target_policy = PPO2.load(
                    load_policy,
                    # env=DummyVecEnv([lambda: env]),
                    # disabled tensorboard temporarily
                    # tensorboard_log='TBlogs/'+self.env_name,
                    verbose=1,
                    n_steps=args['n_steps'],
                    nminibatches=args['nminibatches'],
                    lam=args['lam'],
                    gamma=args['gamma'],
                    noptepochs=args['noptepochs'],
                    ent_coef=args['ent_coef'],
                    learning_rate=args['learning_rate'],
                    cliprange=args['cliprange'],
                )
            elif algo == "TD3":
                args = args['TD3'][self.env_name]

                n_actions = self.sim_env.action_space.shape[-1]
                action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                                 sigma=0.1 * np.ones(n_actions))
                self.target_policy = TD3.load(
                    load_policy,
                    # env=DummyVecEnv([lambda: env]),
                    # tensorboard_log='data/TBlogs/'+self.env_name,
                    verbose=0,
                    batch_size=args['batch_size'],
                    buffer_size=args['buffer_size'],
                    gamma=args['gamma'],
                    gradient_steps=args['gradient_steps'],
                    learning_rate=args['learning_rate'],
                    learning_starts=args['learning_starts'],
                    action_noise=action_noise,
                    train_freq=args['train_freq'],
                )
            elif algo == "TRPO":
                print('Using TRPO as the Target Policy Algo')

                args = args['TRPO'][self.env_name]

                self.target_policy = TRPO.load(
                    load_policy,
                    # env=DummyVecEnv([lambda:env]),
                    verbose=0,
                    # disabled tensorboard temporarily
                    tensorboard_log='data/TBlogs/' + self.env_name if tensorboard else None,
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

            else:
                raise NotImplementedError("Algo " + algo + " not supported yet")

    def _init_normalization_stats(self, training=False):
        print('Using a policy trained in normalized environment.. Loading normalizer')
        self.target_policy_norm_obs = VecNormalize.load('data/models/env_stats/' + self.env_name + '.pkl',
                                                        venv=DummyVecEnv([lambda: gym.make(self.env_name)]))
        self.target_policy_norm_obs.training = training
        self.target_policy_norm_obs.norm_obs = True

    def wgan_loss(self, output, target):
        b_real = output[target == 1.0]
        b_fake = output[target == 0.0]
        rew_bias = -torch.clamp(b_fake, max=0).mean() - torch.clamp(b_real, max=0).mean()
        loss = torch.mean(b_fake) - torch.mean(b_real) + 2 * rew_bias
        return loss

    def logit_bernoulli_entropy(self, logits):
        ent = (1. - torch.nn.Sigmoid()(logits)) * logits - torch.nn.LogSigmoid()(logits)
        return ent

    def bce_with_entropy_loss(self, output, target):
        ent_coeff = 0.00  # not using entropy
        gen_exp_loss = torch.nn.BCEWithLogitsLoss()(output, target)
        if ent_coeff > 0.0:
            entropy = ent_coeff * torch.mean(self.logit_bernoulli_entropy(output))
        else:
            entropy = 0.0
        return torch.mean(gen_exp_loss) - entropy

    def _init_rgat_models(self, algo="TRPO",  # algo is the ATP algo which is ppo in exp
                          ent_coeff=None,
                          max_kl=None,
                          clip_range=None,
                          atp_loss_function='GAIL',
                          disc_lr=3e-4,
                          nminibatches=4,
                          noptepochs=10,
                          atp_lr=3e-4,
                          ):
        """
        Initializes the action transformer policy and discriminator for
        GARAT
        """

        ########### CREATE DISCRIMINATOR ##########

        num_inputs = self.sim_env.action_space.shape[0] * (self.frames)
        num_inputs += self.sim_env.observation_space.shape[0] * (1 + self.frames)
        # input to the discriminator is S_t, a_t, S_t+1
        if atp_loss_function == 'WGAN':
            cprint('USING WGAN FORMULATION. No output activation', 'red', 'on_yellow')

        self.discriminator = Discriminator(
            n_feature=num_inputs,
            n_hidden=64,
            activations=nn.ReLU,
            action_space=self.sim_env.action_space.shape[0]).to(self.device)

        self.discriminator_norm_x = ((np.zeros(num_inputs),
                                      np.ones(num_inputs)), 0)

        if atp_loss_function == 'WGAN':
            self.discriminator_loss = self.wgan_loss
        else:

            self.discriminator_loss = self.bce_with_entropy_loss

        self.optimizer = torch.optim.AdamW(
            self.discriminator.parameters(), lr=disc_lr, weight_decay=1e-3)

        ########### CREATE ACTION TRANSFORMER POLICY ##########
        env = gym.make(self.env_name)
        if self.mujoco_norm: env = MujocoNormalized(env)

        self.atp_environment = ATPEnv(env=env,
                                      target_policy=self.target_policy,
                                      discriminator=self.discriminator,
                                      fwd_model=None,
                                      beta=1.0,
                                      device=self.device,
                                      train_noise=0.0,
                                      loss=atp_loss_function,
                                      normalizer=self.target_policy_norm_obs,
                                      frames=self.frames,
                                      data_collection_mode=False,
                                      expt_path=self.expt_path,
                                      )

        self.atp_environment = DummyVecEnv([lambda: self.atp_environment])
        # self.atp_environment = VecNormalize(self.atp_environment, training=True, norm_obs=True,
        #                                     norm_reward=False)

        if algo == "TD3":
            n_actions = self.atp_environment.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
            self.action_tf_policy = TD3(
                policy=MlpPolicy,
                env=DummyVecEnv([lambda: self.atp_environment]),
                verbose=1,
                # tensorboard_log='data/TBlogs/action_transformer_policy',
                batch_size=2048,
                buffer_size=1000000,
                train_freq=10,
                gradient_steps=1000,
                gamma=0.99,  # NOTE : here, the gamma should be zero !
                learning_rate=0.0003,
                action_noise=action_noise,
                learning_starts=50,
            )
        elif algo == "TRPO":
            self.action_tf_policy = TRPO(
                policy=OtherMlpPolicy,
                env=DummyVecEnv([lambda: self.atp_environment]),
                verbose=0,
                # tensorboard_log='data/TBlogs/action_transformer_policy',
                timesteps_per_batch=self.gsim_trans,  # self.real_trans,
                lam=0.95,
                max_kl=max_kl,
                gamma=0.99,  # NOTE : here, the gamma should be zero !
                vf_iters=1,
                vf_stepsize=atp_lr,
                entcoeff=ent_coeff,
                cg_damping=0.01,
                cg_iters=1
            )
        elif algo == "PPO2":
            class CustomPPO2Policy(FeedForwardPolicy):
                def __init__(self, *args, **kwargs):
                    super(CustomPPO2Policy, self).__init__(*args, **kwargs,
                                                           net_arch=[dict(pi=[64, 64],
                                                                          vf=[64, 64])],
                                                           feature_extraction="mlp")

            # cprint('SINGLE BATCH SIZE : ' + str(self.single_batch_size), 'red', attrs=['blink'])
            self.action_tf_policy = PPO2(
                policy=OtherMlpPolicy,
                # policy = CustomPPO2Policy,
                env=DummyVecEnv([lambda: self.atp_environment]),
                nminibatches=nminibatches,
                n_steps=self.gsim_trans if self.single_batch_size is None else 5000,
                # nminibatches*self.single_batch_size,
                ent_coef=ent_coeff,
                noptepochs=noptepochs,
                lam=0.95,
                cliprange=clip_range,
                learning_rate=atp_lr,
            )

        elif algo == "SAC":
            print('~~ Initializing SAC action transformer policy ~~')
            self.action_tf_policy = SAC(
                policy=SACMlpPolicy,
                env=DummyVecEnv([lambda: self.atp_environment]),
                # tensorboard_log='data/TBlogs/action_transformer_policy',
                verbose=0,
                batch_size=1024,
                buffer_size=1000000)
        else:
            raise NotImplementedError("Algo " + algo + " not supported")

    def train_target_policy_in_grounded_env(self, grounding_step, alpha=1.0,
                                            time_steps=LEARN_TIMESTEPS,
                                            use_eval_callback=False,
                                            save_model=True,
                                            use_deterministic=False):
        """Trains target policy in grounded env"""
        print('TRAINING TARGET POLICY IN GROUNDED ENVIRONMENT FOR ', time_steps, ' TIMESTEPS')
        if use_deterministic:
            print('USING DETERMINISTIC ATP')
        else:
            print('USING STOCHASTIC ATP')

        env = gym.make(self.env_name)
        if self.mujoco_norm: env = MujocoNormalized(env)

        if 'Ant' in self.env_name: use_deterministic = True

        grnd_env = GroundedEnv(env=env,
                               action_tf_policy=self.action_tf_policy,
                               # action_tf_env=self.atp_environment,
                               alpha=alpha,
                               debug_mode=False,
                               normalizer=self.target_policy_norm_obs,
                               data_collection_mode=False,
                               use_deterministic=use_deterministic,
                               atp_policy_noise=0.01 if use_deterministic else 0.0,
                               )

        self.grounded_sim_env = DummyVecEnv([lambda: grnd_env])

        # set target policy in grounded sim env
        self.target_policy.set_env(self.grounded_sim_env)
        cprint('SET THE TARGET POLICY IN GROUNDED ENVIRONMENT', 'red', 'on_green')

        # pylint: disable=unexpected-keyword-arg
        self.target_policy.learn(total_timesteps=time_steps,
                                 reset_num_timesteps=True,
                                 )

        # save the best model
        if save_model:
            if use_eval_callback:
                shutil.move(self.expt_path + '/best_model.zip',
                            self.expt_path + '/target_policy_' + str(grounding_step - 1) + '.pkl')
                self.target_policy.load(self.expt_path + '/target_policy_' + str(grounding_step - 1) + '.pkl')
            else:
                self.target_policy.save(self.expt_path + '/target_policy_' + str(grounding_step - 1) + '.pkl')

    def test_grounded_environment(self,
                                  grounding_step,
                                  alpha=1.0,
                                  random=True):
        """Tests the grounded environment for action transformation"""
        print("TESTING GROUNDED ENVIRONMENT")
        env = gym.make(self.env_name)
        if self.mujoco_norm:
            cprint('Using Custom Mujoco Normalization', 'red', 'on_yellow')
            env = MujocoNormalized(env)

        # if self.target_policy_norm_obs is not None:
        #     cprint('Initializing Normalization Stats', 'red','on_yellow')
        #     self._init_normalization_stats(training=False)

        grnd_env = GroundedEnv(env=env,
                               action_tf_policy=self.action_tf_policy,
                               # action_tf_env=self.atp_environment,
                               alpha=alpha,
                               debug_mode=True,
                               normalizer=self.target_policy_norm_obs,
                               use_deterministic=True,
                               )

        obs = grnd_env.reset()
        time_step_count = 0
        for _ in trange(2048):
            time_step_count += 1
            if not random:
                action, _ = self.target_policy.predict(obs, deterministic=True)
                action += np.random.normal(0, 0.01, action.shape[0])
            else:
                action = self.sim_env.action_space.sample()
            obs, _, done, _ = grnd_env.step(action)
            if done:
                obs = grnd_env.reset()
                done = False

        grnd_env.sim2sim_plot_action_transformation_graph(
            expt_path=self.expt_path,
            grounding_step=grounding_step,
        )
        grnd_env.close()

    def train_action_transformer_policy(self,
                                        beta=0.0,
                                        time_steps=5000,
                                        num_epochs=None,
                                        loss_function='GAIL',
                                        single_batch_test=False,
                                        ):

        if num_epochs is not None:
            time_steps = self.gsim_trans * num_epochs
        print('TRAINING ACTION TRANSFORMER POLICY FOR ', time_steps, ' TIMESTEPS')

        # refresh the discriminator in the ATP Env as the discriminator been trained
        self.atp_environment.env_method("refresh_disc",
                                        target_policy=self.target_policy,
                                        discriminator=self.discriminator,
                                        disc_norm=self.discriminator_norm_x,
                                        )

        self.atp_environment.reset()

        # set action transformer_policy in atp env
        self.action_tf_policy.set_env(self.atp_environment)

        # pylint: disable=unexpected-keyword-arg
        self.action_tf_policy.learn(total_timesteps=time_steps if not single_batch_test else 5000,
                                    # 2*self.single_batch_size,
                                    reset_num_timesteps=False)

    def collect_experience_from_real_env(
            self,
            num_real_traj=None):
        """
        Collects real world experience by deploying target policy on real
        environment
        """
        if num_real_traj is None: num_real_traj = self.NUM_REAL_WORLD_TRAJECTORIES
        print('COLLECTING REAL WORLD TRAJECTORIES')
        # without noise
        # make normalized real env
        if self.target_policy_norm_obs is not None:
            print('loading stats for real env : ', self.real_env_name)
            real_env_for_data_collection = VecNormalize.load('data/models/env_stats/' + self.env_name + '.pkl',
                                                             DummyVecEnv([lambda: gym.make(self.real_env_name)]))
        else:
            if 'Minitaur' in self.real_env_name:
                real_env_for_data_collection = gym.make(self.real_env_name.replace('BulletEnv', 'OnRackBulletEnv'))
                if self.mujoco_norm:
                    cprint('~ Mujoco Normalization called ~', 'red', 'on_yellow')
                    real_env_for_data_collection = MujocoNormalized(real_env_for_data_collection)
                real_env_for_data_collection = DummyVecEnv([lambda: real_env_for_data_collection])
            else:
                real_env_for_data_collection = self.real_env
            # real_env_for_data_collection = self.real_env

        real_Ts = collect_gym_trajectories(env=real_env_for_data_collection,
                                           policy=self.target_policy,
                                           limit_trans_count=int(self.real_trans),
                                           num=None,
                                           add_noise=0.0,  # TODO: check whether add noise will help
                                           deterministic=False,
                                           )

        self.avg_real_traj_length = [np.average([len(real_Ts[z]) for z in range(len(real_Ts))])]
        self.max_real_traj_length = [np.max([len(real_Ts[z]) for z in range(len(real_Ts))])]

        # print('LENGTH OF FIRST TRAJECTORY : ', len(real_Ts[0]))
        # print('AVERAGE LENGTH OF TRAJECTORY : ', self.avg_real_traj_length)
        # print('MAX LENGTH OF TRAJECTORY : ', self.max_real_traj_length)

        X_list, Y_list = [], []
        for T in real_Ts:  # For each trajectory:
            for i in range(len(T) - self.frames):
                X = np.array([])
                # Append previous self.frames states S_t
                for j in range(self.frames):
                    X = np.append(X, T[i + j][0])  # state
                    # append actions
                    X = np.append(X, T[i + j][1])  # action

                # Append next state S_{t+1}
                X = np.append(X, T[i + self.frames][0])
                X_list.append(X)

                # Append label = real
                Y_list.append(np.array([1.0]))

        self.real_X_list = X_list  # [num x transition(s,a,s_)]
        self.real_Y_list = Y_list  # [num x 1]

    def train_discriminator(self,
                            iter_step,
                            grounding_step,
                            num_sim_traj=None,
                            num_epochs=MAX_EPOCHS,
                            inject_instance_noise=False,
                            compute_grad_penalty=True,
                            nminibatches=4,
                            single_batch_test=False,
                            debug_discriminator=True,
                            ):
        """
        Trains the discriminator function that classifies real and fake
        trajectories
        """

        # if compute_grad_penalty: cprint('COMPUTING GRAD PENALTY', 'yellow', attrs=['blink'])

        if num_sim_traj is None: num_sim_traj = self.NUM_SIM_WORLD_TRAJECTORIES
        X_list = []  # previous states + action + next state
        Y_list = []  # label for the trajectory : real:[1] / fake:[0]

        ######### COLLECT REAL TRAJECTORIES ###################
        # load the collected real trajectories
        X_list.extend(self.real_X_list)
        Y_list.extend(self.real_Y_list)

        # randomly subsample
        if single_batch_test:
            assert len(X_list) >= self.single_batch_size, "Using too less data"
            X_list_indices = random.sample(np.arange(len(X_list)).tolist(), k=self.single_batch_size)
            X_list = [X_list[i] for i in X_list_indices]
            Y_list = [Y_list[i] for i in X_list_indices]

        # print('Real data transition count : ', len(Y_list))  # (s,a,s_)

        ######### COLLECT FAKE TRAJECTORIES ###################

        # create the grounded environment to collect fake trajectories from
        if self.target_policy_norm_obs is not None:
            self._init_normalization_stats(training=False)

        if 'Minitaur' in self.real_env_name:
            env = gym.make(self.env_name.replace('BulletEnv', 'OnRackBulletEnv'))
            if self.mujoco_norm: env = MujocoNormalized(env)
        else:
            env = gym.make(self.env_name)
            if self.mujoco_norm: env = MujocoNormalized(env)
        # env = gym.make(self.env_name)
        # if self.mujoco_norm: env = MujocoNormalized(env)

        grnd_env = GroundedEnv(env=env,
                               action_tf_policy=self.action_tf_policy,
                               # action_tf_env=self.atp_environment,
                               debug_mode=False,
                               normalizer=self.target_policy_norm_obs,
                               use_deterministic=True,
                               atp_policy_noise=0.0,
                               )

        grnd_env = DummyVecEnv([lambda: grnd_env])

        fake_Ts = collect_gym_trajectories(
            env=grnd_env,
            policy=self.target_policy,
            num=int(num_sim_traj),
            add_noise=0.0,
            limit_trans_count=5000 if single_batch_test else int(self.real_trans),  # -int(self.gsim_trans),
            max_timesteps=self.max_real_traj_length[0],
            deterministic=False,  # TODO: check here whether need to be set as deterministic
        )

        # print('LENGTH OF FIRST TRAJECTORY : ', len(fake_Ts[0]))
        # print('NUM TRAJS : ', len(fake_Ts))
        # print('AVERAGE LENGTH OF TRAJECTORY : ', [np.average([len(fake_Ts[z]) for z in range(len(fake_Ts))])])
        # print('MAX TRAJECTORY LENGTH : ', [np.max([len(fake_Ts[z]) for z in range(len(fake_Ts))])])

        # unpack trajectories and create the dataset to train discriminator

        X_list_fake, Y_list_fake = [], []

        for T in fake_Ts:  # For each trajectory:
            for i in range(len(T) - self.frames):
                X = np.array([])
                # Append previous self.frames states S_t
                for j in range(self.frames):
                    X = np.append(X, T[i + j][0])
                    # Fixing a bug : adding actions to the frames
                    X = np.append(X, T[i + j][1])

                # Append action a_t
                # X = np.append(X, T[i + self.frames - 1][1])

                # Append next state S_{t+1}
                X = np.append(X, T[i + self.frames][0])
                X_list_fake.append(X)  # shape it into [s_t,a_t,s_t+1]

                # Append label = fake
                Y_list_fake.append(np.array([0.0]))

        if single_batch_test:
            assert len(X_list) >= self.single_batch_size, "Using too less data"
            X_list_indices = random.sample(np.arange(len(X_list_fake)).tolist(), k=self.single_batch_size)
            X_list_fake = [X_list_fake[i] for i in X_list_indices]
            Y_list_fake = [Y_list_fake[i] for i in X_list_indices]
            X_list.extend(X_list_fake)
            Y_list.extend(Y_list_fake)
        else:
            X_list.extend(X_list_fake)
            Y_list.extend(Y_list_fake)

        # print('STARTING TO TRAIN THE DISCRIMINATOR')
        #
        # print('Num disc updates : ', len(Y_list) / self.single_batch_size)

        self.discriminator_norm_x, _ = train_model_es(
            model=self.discriminator,
            x_list=X_list,
            y_list=Y_list,
            optimizer=self.optimizer,
            criterion=self.discriminator_loss,
            problem='classification',
            max_epochs=num_epochs if not single_batch_test else 1,
            checkpoint_name=self.expt_label,
            num_cores=self.num_cores,
            label=str(grounding_step) + '_' + str(iter_step),
            device=self.device,
            normalize_data=True,
            use_es=False,
            dump_loss_to_file=debug_discriminator,
            expt_path=self.expt_path,
            compute_grad_penalty=compute_grad_penalty,
            batch_size=int(len(Y_list) / nminibatches) if not single_batch_test else self.single_batch_size,
        )

        # set discriminator to eval mode after training
        self.discriminator = self.discriminator.eval()
        self.discriminator = self.discriminator.to(self.device)

    def save_target_policy(self, iter_num=None):
        """Saves target policy"""
        # April 19, 2020 : Currently not using this method to save the target policy
        print('~~~ SAVING TARGET POLICY ~~~')
        iter_path = '' if iter_num is None else '_' + str(iter_num)

        self.target_policy.save(self.expt_path + '/target_policy' +
                                iter_path + '.pkl')

        print('~~~~~~~~~~ SAVED MODELS ! ~~~~~~')

    def save_atp(self, grounding_step=None):
        """Saves the action transformer policy"""
        print('##### SAVING ACTION TRANSFORMER POLICY #####')
        # iter_path = '' if iter_num is None else '_'+str(iter_num)

        self.action_tf_policy.save(self.expt_path + '/action_transformer_policy'
                                   + grounding_step + '.pkl')
        # self.atp_environment.save(self.expt_path+'/atp_env'+grounding_step+'.pkl')

        print('##### SAVED ATP MODEL ! #####')

    def save_grounded_env(self, grounding_step=None):
        with open(self.expt_path + '/grnd_env_' + str(grounding_step) + '.pkl', "wb") as pickle_file:
            pickle.dump(self.grounded_sim_env, pickle_file)

    def load_model(self, model_path):
        """Loads the model for the target policy from disk"""
        self.target_policy.load(model_path)
