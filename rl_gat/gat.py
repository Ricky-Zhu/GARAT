"""Implements GAT algorithm

Implementation of Grounded Action Transformation as described in AAAI-17 paper
for OpenAI Gym environments.
"""
# pylint: disable=invalid-name,too-many-locals, wrong-import-position, too-many-arguments, not-callable
# Suppress these warnings to allow for certain machine learning conventions
import random
import os
import gym, math
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines import PPO2, TD3
import numpy as np
import torch, torchvision

torch.backends.cudnn.deterministic = True
is_torchvision_installed = True
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from torch import autograd

# The number of frames to feed into the action transformer
NUM_FRAMES_INPUT = 1
# Number of CPU Cores
NUM_CORES = 8
# Number of trajectories to collect on 'real' environment
NUM_REAL_WORLD_TRAJECTORIES = int(500)
# Number of trajectories to collect on simulated environment
NUM_SIM_WORLD_TRAJECTORIES = int(500)
# Max number of epochs to train models
MAX_EPOCHS = 1000
# Fraction of dataset to use as validation
VALIDATION_SIZE = 0.2
# No. of timesteps to learn for
LEARN_TIMESTEPS = int(1000000 * 0.01)
# LEARN_TIMESTEPS = 5000

# No. of threads to used during RL step
NUM_RL_THREADS = 1


class NoisyRealEnv(gym.ActionWrapper):
    def __init__(self, env, noise_value=0.1):
        super(NoisyRealEnv, self).__init__(env)
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.action_shape = self.action_space.shape[0]
        self.noise_value = noise_value

    def action(self, action):
        action += np.random.normal(loc=0,
                                   scale=self.noise_value,
                                   size=self.action_shape)
        return action.clip(self.low, self.high)


class Net(torch.nn.Module):
    """network that defines the forward dynamics model"""

    def __init__(self, n_feature, n_hidden, n_output, activations=nn.Tanh, action_activation=False):
        super(Net, self).__init__()

        self.fc_in = nn.Linear(n_feature, n_hidden)
        self.fc_h1 = nn.Linear(n_hidden, n_hidden)
        self.fc_h2 = nn.Linear(n_hidden, n_hidden)

        self.dropout = nn.Dropout(0.5)

        self.fc_out = nn.Linear(n_hidden, n_output)

        self.single_fc = nn.Linear(n_feature, n_output)
        self.last_activation = action_activation
        self.activations = activations

        torch.nn.init.xavier_uniform_(self.fc_in.weight)
        torch.nn.init.xavier_uniform_(self.fc_h1.weight)
        torch.nn.init.xavier_uniform_(self.fc_h2.weight)

        torch.nn.init.xavier_uniform_(self.fc_out.weight)
        torch.nn.init.xavier_uniform_(self.single_fc.weight)

    # pylint: disable=arguments-differ
    def forward(self, x):
        out = self.activations()(self.fc_in(x))
        out = self.activations()(self.fc_h1(out))
        out = self.dropout(out)
        out = self.activations()(self.fc_h2(out))
        out = self.fc_out(out)

        skip = self.single_fc(x)

        if self.last_activation:
            return nn.Tanh()(out)  # using a tanh activation for actions

        return out + skip


class ActionTransformer:
    """Combination of a forward and inverse model"""

    def __init__(self,
                 forward_model,
                 inverse_model,
                 forward_norms=((0., 1.), (0., 1.)),
                 inverse_norms=((0., 1.), (0., 1.)),
                 frames=NUM_FRAMES_INPUT,
                 alpha=0.5):
        self.fwd_model = forward_model
        self.fwd_model.eval()
        self.inv_model = inverse_model
        self.inv_model.eval()

        self.fwd_norms = forward_norms
        self.inv_norms = inverse_norms
        self.frames = frames
        self.alpha = alpha

    def transform(self, action, prev_obs, obs):
        """Transforms an action based on observation history"""
        # Concatenate state-action pair
        concat_sa = np.array([])
        for pobs in prev_obs[1:]:
            concat_sa = np.append(concat_sa, pobs)
        concat_sa = np.append(concat_sa, obs)
        concat_sa = np.append(concat_sa, action)

        # Normalize
        concat_sa = apply_norm(concat_sa, self.fwd_norms[0])

        # Convert to tensor
        concat_sa = torch.tensor(concat_sa).float().to('cpu')
        with torch.no_grad():
            # Apply forward model
            d_next_state = self.fwd_model.forward(concat_sa)
            d_next_state = d_next_state.detach().cpu().numpy()
            # Un-normalize
            d_next_state = unapply_norm(d_next_state, self.fwd_norms[1])

            # Concatenate state, predicted next_state
            concat_ss = np.array([])
            for pobs in prev_obs[1:]:
                concat_ss = np.append(concat_ss, pobs)
            concat_ss = np.append(concat_ss, obs)
            concat_ss = np.append(concat_ss, d_next_state)
            # Normalize
            concat_ss = apply_norm(concat_ss, self.inv_norms[0])
            # Convert to tensor
            concat_ss = torch.tensor(concat_ss).float().to('cpu')
            # Apply inverse model
            tf_action = self.inv_model.forward(concat_ss)
            tf_action = tf_action.detach().cpu().numpy()
            # Un-normalize
            tf_action = unapply_norm(tf_action, self.inv_norms[1])
            # Apply alpha term
            tf_action = (1.0 - self.alpha) * action + self.alpha * tf_action

        return tf_action


def apply_norm(dataset, norm):
    """Normalizes data given a (mean, std) tuple"""
    return (dataset - norm[0]) / (norm[1] + 1e-8)


def unapply_norm(dataset, norm):
    """Inverse operation of _apply_norm"""
    return (dataset * norm[1]) + norm[0]


#
# class GroundedEnv(gym.ActionWrapper):
#     """Creates grounded environment from a base env and an action transformer"""
#     #pylint: disable=abstract-method
#     def __init__(self, env, grounding_step_number, action_transformer=None, debug_mode=True):
#         super(GroundedEnv, self).__init__(env)
#         self.act_tf = action_transformer
#         self.prev_frames = []
#         # Step needs to be called for there to be a latest_obs
#         self.latest_obs = None
#         # List to store trajectory
#         # self.T = []
#         # self.Ts = []
#         self.time_step_counter = 0
#         self.debug_mode = debug_mode
#         self.grounding_step_number = grounding_step_number
#         if debug_mode:
#             self.transformed_action_list = []
#             self.raw_actions_list = []
#
#     def reset(self, **kwargs):
#         self.latest_obs = self.env.reset(**kwargs)
#         # self.prev_frames = [self.latest_obs for _ in range(self.act_tf.frames)]
#         self.prev_frames = [self.latest_obs]
#         # self.T = []
#         # reset time
#         self.time_step_counter = 0
#         return self.latest_obs
#
#     def step(self, action):
#         """record latest observation while calling env.step"""
#
#         # first increment self.time_step
#         self.time_step_counter += 1
#
#         transformed_action = self.act_tf.transform(action,
#                                                    self.prev_frames,
#                                                    self.latest_obs)
#
#         # if self.debug_mode and self.time_step_counter <= 1e4:
#         #     self.transformed_action_list.append(transformed_action)
#         #     self.raw_actions_list.append(action)
#
#         # self.T.append((self.latest_obs, transformed_action))
#         self.latest_obs, rew, done, info = self.env.step(transformed_action)
#         self.prev_frames = self.prev_frames[1:]+[self.latest_obs]
#
#         # if done:
#         #     self.T.append((self.latest_obs, None))
#         #     self.Ts.extend([self.T])
#
#         return self.latest_obs, rew, done, info
#
#     def sim2sim_plot_action_transformation_graph(self):
#         plt.figure(figsize=(10, 10))
#         self.raw_actions_list = np.asarray(self.raw_actions_list)
#         self.transformed_action_list = np.asarray(self.transformed_action_list)
#         plt.plot(self.raw_actions_list[:10000, 0], self.transformed_action_list[:10000, 0], 'bo',
#                  self.raw_actions_list[:10000, 1], self.transformed_action_list[:10000, 1], 'go',
#                  self.raw_actions_list[:10000, 2], self.transformed_action_list[:10000, 2], 'ro',
#                  [-1, 1], [-1, 1], 'k-')
#         plt.xlabel('Input Actions')
#         plt.ylabel('Transformed Actions')
#         plt.savefig(os.getcwd()+'/data/models/figs/'+str(self.grounding_step_number)+'_e4_.png')
#         plt.show()
#         plt.clf()
#         plt.plot(self.raw_actions_list[:1000, 0], self.transformed_action_list[:1000, 0], 'bo',
#                  self.raw_actions_list[:1000, 1], self.transformed_action_list[:1000, 1], 'go',
#                  self.raw_actions_list[:1000, 2], self.transformed_action_list[:1000, 2], 'ro',
#                  [-1, 1], [-1, 1], 'k-')
#         plt.xlabel('Input Actions')
#         plt.ylabel('Transformed Actions')
#         plt.savefig(os.getcwd()+'/data/models/figs/'+str(self.grounding_step_number)+'_e3_.png')
#
#     def close(self):
#         """save all trajectories recorded so far to disk"""
#         # if os.path.isfile('./data/tmp/trajectories.npy'):
#         #     tmp_var = np.load('./data/tmp/trajectories.npy')
#         #     self.Ts = np.append(tmp_var, np.array(self.Ts), axis=0)
#
#         # dont append. Just save collected trajectories.
#         # np.save('./data/tmp/trajectories', self.Ts)
#         # np.save('./data/tmp/transformed_actions', self.transformed_action_list)
#         # np.save('./data/tmp/raw_actions', self.raw_actions_list)
#         self.env.close()


class GroundedEnv(gym.ActionWrapper):
    """Creates grounded environment from a base env and an action transformer"""

    # pylint: disable=abstract-method
    def __init__(self, env, action_transformer=None, grounding_step_number=0):
        super(GroundedEnv, self).__init__(env)
        self.grounding_step_number = grounding_step_number
        self.act_tf = action_transformer
        # Step needs to be called for there to be a latest_obs
        self.latest_obs = None
        # List to store trajectory
        self.T = []
        self.Ts = []
        self.transformed_action_list = []
        self.raw_actions_list = []

    def reset(self, **kwargs):
        self.latest_obs = self.env.reset(**kwargs)
        self.T = []
        return self.latest_obs

    def step(self, action):
        """record latest observation while calling env.step"""
        transformed_action = self.action(action)
        self.transformed_action_list.append(transformed_action)
        self.raw_actions_list.append(action)
        self.T.append((self.latest_obs, transformed_action))
        self.latest_obs, rew, done, info = self.env.step(transformed_action)

        if done:
            self.T.append((self.latest_obs, None))
            self.Ts.extend([self.T])
            self.T = []  # reset self.T

        return self.latest_obs, rew, done, info

    def close(self):
        """save all trajectories recorded so far to disk"""
        # if os.path.isfile('./data/tmp/trajectories.npy'):
        #     tmp_var = np.load('./data/tmp/trajectories.npy')
        #     self.Ts = np.append(tmp_var, np.array(self.Ts), axis=0)

        # dont append. Just save collected trajectories.
        # np.save('./data/tmp/trajectories', self.Ts)
        # np.save('./data/tmp/transformed_actions', self.transformed_action_list)
        # np.save('./data/tmp/raw_actions', self.raw_actions_list)
        self.env.close()

    def sim2sim_plot_action_transformation_graph(self, show_plot=False, plot_e3=False, expt_label=None):
        num_action_space = self.env.action_space.shape[0]
        action_low = self.env.action_space.low[0]
        action_high = self.env.action_space.high[0]

        self.raw_actions_list = np.asarray(self.raw_actions_list)
        self.transformed_action_list = np.asarray(self.transformed_action_list)
        colors = ['bo', 'go', 'ro', 'mo', 'yo', 'ko']

        if num_action_space > len(colors):
            print("Unsupported Action space shape.")
            return

        # plotting the data points starts here
        fig = plt.figure(figsize=(int(10 * num_action_space), 10))
        for act_num in range(num_action_space):
            ax = fig.add_subplot(1, num_action_space, act_num + 1)
            ax.plot(self.raw_actions_list[:10000, act_num], self.transformed_action_list[:10000, act_num],
                    colors[act_num])
            ax.plot([action_low, action_high], [action_low, action_high], 'k-')  # black line
            ax.title.set_text('Action Space # :' + str(act_num + 1) + '/' + str(num_action_space))
            ax.set(xlabel='Input Actions', ylabel='Transformed Actions', xlim=[action_low, action_high],
                   ylim=[action_low, action_high])
        plt.suptitle('Plot of Input action $a_t$ vs Transformed action $\\tilde{a}_t$')
        plt.savefig(os.getcwd() + '/data/models/gat/' + expt_label + '/' + str(self.grounding_step_number) + '_e4_.png')
        if show_plot: plt.show()
        plt.close()

        if not plot_e3: return

        fig = plt.figure(figsize=(int(10 * num_action_space), 10))
        for act_num in range(num_action_space):
            ax = fig.add_subplot(1, num_action_space, act_num + 1)
            ax.plot(self.raw_actions_list[:1000, act_num], self.transformed_action_list[:1000, act_num],
                    colors[act_num])
            ax.plot([action_low, action_high], [action_low, action_high], 'k-')  # black line
            ax.title.set_text('Action Space # :' + str(act_num + 1) + '/' + str(num_action_space))
            ax.set(xlabel='Input Actions', ylabel='Transformed Actions', xlim=[action_low, action_high],
                   ylim=[action_low, action_high])
        plt.suptitle('Plot of Input action $a_t$ vs Transformed action $\\tilde{a}_t$')
        plt.savefig(os.getcwd() + '/data/models/gat/' + expt_label + '/' + str(self.grounding_step_number) + '_e3_.png')
        plt.close()

    def action(self, action):
        # Concatenate state-action pair
        concat_sa = np.hstack((self.latest_obs, action))

        # Normalize
        concat_sa = apply_norm(concat_sa, self.act_tf.fwd_norms[0])
        # Convert to tensor
        concat_sa = torch.tensor(concat_sa).float().to('cpu')
        with torch.no_grad():
            # Apply forward model
            d_next_state = self.act_tf.fwd_model.forward(concat_sa)
            d_next_state = d_next_state.detach().cpu().numpy()
            # Un-normalize
            d_next_state = unapply_norm(d_next_state, self.act_tf.fwd_norms[1])

            # Concatenate state, predicted next_state
            concat_ss = np.hstack((self.latest_obs, d_next_state))
            # Normalize
            concat_ss = apply_norm(concat_ss, self.act_tf.inv_norms[0])
            # Convert to tensor
            concat_ss = torch.tensor(concat_ss).float().to('cpu')
            # Apply inverse model
            tf_action = self.act_tf.inv_model.forward(concat_ss)
            tf_action = tf_action.detach().cpu().numpy()
            # Un-normalize
            tf_action = unapply_norm(tf_action, self.act_tf.inv_norms[1])
            # Apply alpha term
            tf_action = (1 - self.act_tf.alpha) * action + self.act_tf.alpha * tf_action

        return tf_action


def collect_gym_trajectories(
        env,
        policy,
        num,
        max_timesteps=10000,
        collect_rew=False,
        add_noise=0.0,
        deterministic=True,
        limit_trans_count=None,
):
    """Generates trajectories from an environment

    :param env: gym environment that generates trajectories
    :param policy: behavior policy
    :type policy: PPO2
    :param num: number of trajectories to collect
    :type num: int
    :return: List of trajectories, which are lists of (s,a) tuples
    :rtype: list
    """
    Ts = []  # Initialize list of trajectories
    if collect_rew:
        Rs = []  # Initialize list of rewards

    num_env = env.num_envs  # Number of parallel envs to use
    action_lim = abs(env.action_space.high)

    transition_count = 0
    # for _ in tqdm(range(num//num_env)):
    while True:
        # Done indicates whether we are in a terminal state
        done = [False for i in range(num_env)]
        # Finished indicates whether we have ever seen a terminal state
        finished = [False for i in range(num_env)]
        # Create an empty list for each parallel environment
        T = [[] for i in range(num_env)]
        if collect_rew:
            R = [[] for i in range(num_env)]
            rew = None

        # Reset all environments
        obs = env.reset()

        time_steps = 0
        while not all(finished):
            action, _ = policy.predict(obs, deterministic=deterministic)
            action = action + np.random.normal(0, 1, env.action_space.shape[0]) * add_noise
            # clip action within range
            action = np.clip(action, -action_lim, action_lim)
            for e in range(num_env):
                if not finished[e]:
                    if not done[e]:
                        # Append state-action pair
                        T[e].append((obs[e], action[e]))
                        if collect_rew: R[e].append(rew)
                    else:
                        # Append terminal state
                        T[e].append((obs[e], None))
                        if collect_rew: R[e].append(rew)
                        finished[e] = True
            obs, rew, done, _ = env.step(action)
            time_steps += 1
            if limit_trans_count is not None: transition_count += num_env
            # break if policy is running for long on env
            if time_steps >= max_timesteps: break

        Ts.extend(T)
        if collect_rew: Rs.extend(R)

        if limit_trans_count is not None:
            if transition_count > limit_trans_count:
                break

    if collect_rew: return Ts, Rs
    return Ts


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max * len(self.keys)


def compute_grad_pen(x_data, y_data, model):
    expert_data = x_data[(y_data.squeeze() == 1.0)]
    policy_data = x_data[(y_data.squeeze() == 0.0)]

    alpha = torch.randn_like(expert_data).to('cpu')
    mixup_data = alpha * expert_data + (1 - alpha) * policy_data
    mixup_data.requires_grad = True

    disc = model(mixup_data)
    ones = torch.ones(disc.size()).to('cpu')

    grad = autograd.grad(
        outputs=disc,
        inputs=mixup_data,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    grad_pen = 10 * (grad.norm(2, dim=1) - 1).pow(2).mean()
    return grad_pen


def train_model_es(model,
                   x_list,
                   y_list,
                   optimizer,
                   label,
                   criterion=torch.nn.MSELoss(),
                   max_epochs=MAX_EPOCHS,
                   v_size=VALIDATION_SIZE,
                   patience=10,
                   batch_size=128,
                   problem='regression',
                   checkpoint_name='checkpoint',
                   num_cores=NUM_CORES,
                   clip_weights=False,
                   device='cuda',
                   normalize_data=False,
                   use_es=True,
                   dump_loss_to_file=True,
                   expt_path=None,
                   compute_grad_penalty=True,
                   ):
    """Trains a model from given labels with early stopping

    :param model: the model to be trained
    :type model: torch.nn.module
    :param x_list: list of features (same size as y_list)
    :type x_list: list
    :param y_list: list of labels (same size as x_list)
    :type y_list: list
    :param optimizer: optimizer
    :type optimizer: optimizer
    :param criterion: criterion
    :type criterion: criterion
    :param max_epochs: maximum number of epochs
    :type max_epochs: int
    :param v_size: size of validation set [0,1)
    :type v_size: float
    :param patience: number of epochs to wait before early stopping
    :type patience: int
    """
    # for visualizing plots on tensorboard
    # writer = SummaryWriter(log_dir='runs/'+checkpoint_name)

    if not use_es: patience = max_epochs

    valid_size = int(np.floor(len(x_list) * v_size))
    train_size = len(x_list) - valid_size

    # Convert to nd-array
    x_list = np.array(x_list).copy()
    y_list = np.array(y_list).copy()

    class_sample_count = np.array([len(np.where(y_list == t)[0]) for t in np.unique(y_list)])
    # print("real_fake_class_count : ", class_sample_count.shape, class_sample_count)

    if use_es:
        indices = np.random.permutation(x_list.shape[0])
        train_idx, valid_idx = indices[:train_size], indices[train_size:]
        x_train, x_valid = x_list[train_idx, :], x_list[valid_idx, :]
        y_train, y_valid = y_list[train_idx, :], y_list[valid_idx, :]

    else:
        x_train, y_train = x_list, y_list

    # calculate the norm tuple for input and output
    norms_x = (np.mean(x_train, axis=0), np.std(x_train, axis=0))
    norms_y = (np.mean(y_train, axis=0), np.std(y_train, axis=0))

    if not normalize_data:
        norms_x = (np.zeros_like(norms_x[0]), np.ones_like(norms_x[1]))
        norms_y = (np.zeros_like(norms_y[0]), np.ones_like(norms_y[1]))

    # convert to tensors
    x_train_normalized = apply_norm(x_train, norms_x)
    x_train_normalized = torch.Tensor(x_train_normalized)
    if use_es:
        x_valid_normalized = apply_norm(x_valid, norms_x)
        x_valid_normalized = torch.tensor(x_valid_normalized).float()

    if problem == 'classification':
        # when solving a classification problem
        y_train_normalized = torch.Tensor(y_train)
        if use_es: y_valid_normalized = torch.Tensor(y_valid)
    else:
        y_train_normalized = apply_norm(y_train, norms_y)
        y_train_normalized = torch.tensor(y_train_normalized).float()
        if use_es:
            y_valid_normalized = apply_norm(y_valid, norms_y)
            y_valid_normalized = torch.tensor(y_valid_normalized).float()

    train_data = data.TensorDataset(x_train_normalized, y_train_normalized)
    if use_es:
        valid_data = data.TensorDataset(x_valid_normalized, y_valid_normalized)

    # print('total data points : ', np.sum(class_sample_count))

    batch_size = math.floor(batch_size / 2.0) * 2  # round batchsize to the nearest even number
    # print('BATCH SIZE : ', batch_size)
    drop_bool = np.floor(np.sum(class_sample_count) / batch_size) == np.sum(class_sample_count) / batch_size
    # print('drop_last : ', not drop_bool)

    train_loader = data.DataLoader(
        dataset=train_data,
        # shuffle=True,
        batch_size=batch_size,
        num_workers=0,
        sampler=BalancedBatchSampler(train_data, labels=y_train),
        drop_last=not drop_bool)

    if use_es:
        valid_loader = data.DataLoader(
            dataset=valid_data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=0)

    # train the model
    epoch = 0
    best_epoch = 0
    best_loss = None

    while (epoch < max_epochs and
           epoch < best_epoch + patience):
        epoch += 1
        train_losses = []
        if use_es:
            valid_losses = []

        # Train for one epoch
        model = model.train()  # prep for training
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model.forward(x_batch)
            loss = criterion(y_pred, y_batch)

            if compute_grad_penalty:
                grad_pen = compute_grad_pen(x_batch, y_batch, model)
                loss = loss + grad_pen

            # print('\repoch: ' + str(epoch) + ' loss: ' + str(loss.item()), end='')
            # Zero the gradients
            optimizer.zero_grad()
            # perform a backward pass (backpropagation)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate model
        if use_es:
            model = model.eval()  # prep for evaluation
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model.forward(x_batch)
                loss = criterion(y_pred, y_batch)
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)


        if use_es:
            valid_loss = np.average(valid_losses)
            print('validation loss:', valid_loss)
            if best_loss is None or valid_loss < best_loss:
                best_epoch = epoch
                best_loss = valid_loss
                torch.save(model.state_dict(), checkpoint_name + '.pt')
            print('current best:', best_loss, '(epoch', best_epoch, ')')
    # print('\ntraining loss:', train_loss)

    if dump_loss_to_file:
        with open(expt_path + '/disc_loss.txt', "a") as txt_file:
            print(train_loss, file=txt_file)

    if use_es:
        model.load_state_dict(torch.load(checkpoint_name + '.pt'))
        # delete the checkpoint file from memory
        os.remove(checkpoint_name + '.pt')
        print('~~~~~~~~~~ Checkpoint file cleared after training ~~~~~~~~')

    return norms_x, norms_y


class GAT:
    """Implements GAT"""

    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 sim_env_name='Hopper-v2',
                 real_env_name='HopperModified-v2',
                 frames=NUM_FRAMES_INPUT,
                 num_cores=NUM_CORES,
                 num_rl_threads=NUM_RL_THREADS,
                 load_policy=None,
                 algo=None):
        self.env_name = sim_env_name
        self.real_env_name = real_env_name
        self.frames = frames
        self.num_cores = num_cores
        self.fwd_norms_x = (0., 1.)
        self.fwd_norms_y = (0., 1.)
        self.inv_norms_x = (0., 1.)
        self.inv_norms_y = (0., 1.)
        self.num_rl_threads = num_rl_threads
        self.real_env = SubprocVecEnv(
            [lambda: gym.make(self.real_env_name) for i in range(self.num_cores)])
        print('MODIFIED ENV BODY_MASS : ',
              gym.make(self.real_env_name).model.body_mass)
        self.sim_env = SubprocVecEnv(
            [lambda: gym.make(self.env_name) for i in range(self.num_cores)])
        print('SIMULATED ENV BODY_MASS : ',
              gym.make(self.env_name).model.body_mass)

        # lists to reuse experience from previous grounding steps
        self.fwd_model_x_list = []
        self.fwd_model_y_list = []
        self.inv_model_x_list = []
        self.inv_model_y_list = []

        # initialize target policy
        if load_policy is None:
            print('LOADING -RANDOM- INITIAL POLICY')
            self.target_policy = PPO2(
                MlpPolicy,
                env=self.sim_env,
                verbose=1,
                tensorboard_log='data/TBlogs/' + self.env_name)
        else:
            print('LOADING -PRETRAINED- INITIAL POLICY')
            # self.target_policy = SAC.load(
            #     load_policy,
            #     env=SubprocVecEnv([lambda: gym.make(self.env_name)]),
            #     tensorboard_log='data/TBlogs/'+self.env_name,
            #     verbose=1,
            #     batch_size=256,
            #     buffer_size=1000000,
            # )
            # TODO: write easy way to switch algorithms
            # self.target_policy = PPO2.load(
            #         load_policy,
            #         env=SubprocVecEnv([lambda: gym.make(self.env_name)]),
            #         tensorboard_log='TBlogs/'+self.env_name,
            #         verbose=1,
            #         n_steps=256,
            #         # buffer_size=1000000,
            #     )

            n_actions = self.sim_env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            self.target_policy = TD3.load(
                load_policy,
                env=DummyVecEnv([lambda: gym.make(self.env_name)]),
                tensorboard_log='data/TBlogs/' + self.env_name,
                verbose=1,
                batch_size=128,
                gamma=0.99,
                learning_rate=0.001,
                action_noise=action_noise,
                buffer_size=1000000,
            )

        # define the Grounded Action Transformer models here
        self._init_gat_models()
        self.grounded_sim_env = None

    def _init_gat_models(self):
        """Initializes the forward and inverse models for GAT"""
        # define the forward dynamics model and training apparatus here
        num_inputs = self.sim_env.action_space.shape[0]
        num_inputs += self.sim_env.observation_space.shape[0] * self.frames
        self.forward_model = Net(
            n_feature=num_inputs,
            n_hidden=128 * self.frames,
            activations=nn.Tanh,
            n_output=self.sim_env.observation_space.shape[0]).cuda()

        self.forward_model_criterion = torch.nn.MSELoss()
        self.forward_model_optimizer = torch.optim.Adam(
            self.forward_model.parameters(), lr=0.01, weight_decay=0.01)

        # define the inverse dynamics model and training apparatus here :
        num_inputs = (1 + self.frames) * self.sim_env.observation_space.shape[0]
        self.inverse_model = Net(
            n_feature=num_inputs,
            n_hidden=128 * self.frames,
            activations=nn.Tanh,
            n_output=self.sim_env.action_space.shape[0]).cuda()

        self.inverse_model_criterion = torch.nn.MSELoss()
        self.inverse_model_optimizer = torch.optim.Adam(
            self.inverse_model.parameters(), lr=0.01, weight_decay=0.01)

    def test_grounded_env(self, alpha, grounding_step):
        """tests the grounded environment"""
        action_tf = ActionTransformer(
            self.forward_model,
            self.inverse_model,
            (self.fwd_norms_x, self.fwd_norms_y),
            (self.inv_norms_x, self.inv_norms_y),
            self.frames,
            alpha
        )
        # ground the sim environment here
        grnd_env = GroundedEnv(gym.make(self.env_name),
                               action_transformer=action_tf,
                               grounding_step_number=grounding_step)
        grnd_env.reset()
        time_step_count = 0
        for _ in trange(11000):
            time_step_count += 1
            sampled_action = grnd_env.action_space.sample()
            _, _, done, _ = grnd_env.step(sampled_action)
            if done:
                grnd_env.reset()
                done = False

        grnd_env.sim2sim_plot_action_transformation_graph()
        grnd_env.close()

    def train_target_policy_in_grounded_env(self, alpha, grounding_step):
        """Trains target policy in grounded env"""
        action_tf = ActionTransformer(
            self.forward_model,
            self.inverse_model,
            (self.fwd_norms_x, self.fwd_norms_y),
            (self.inv_norms_x, self.inv_norms_y),
            self.frames,
            alpha
        )

        # ground the sim environment here
        grnd_env = GroundedEnv(gym.make(self.env_name),
                               action_transformer=action_tf,
                               grounding_step_number=grounding_step)
        # self.grounded_sim_env = SubprocVecEnv(
        #     [lambda: grnd_env for i in range(self.num_rl_threads)])
        self.grounded_sim_env = DummyVecEnv([lambda: grnd_env])

        # set target policy in grounded sim env
        self.target_policy.set_env(self.grounded_sim_env)

        # pylint: disable=unexpected-keyword-arg
        self.target_policy.learn(total_timesteps=LEARN_TIMESTEPS,
                                 reset_num_timesteps=False)

        # save all trajectories to memory
        # close the grounded env to make space in GPU for further training
        grnd_env.sim2sim_plot_action_transformation_graph()
        self.grounded_sim_env.close()

    def train_forward_model(self, num_traj=NUM_REAL_WORLD_TRAJECTORIES):
        """Trains the forward model based on 'real world' trajectories"""
        print('TRAINING THE FORWARD DYNAMICS MODEL')
        # set the target policy in real environment
        # self.target_policy.set_env(self.real_env)

        # collect experience from real world
        # Ts = collect_gym_trajectories(
        #     self.real_env,
        #     self.target_policy,
        #     num_traj)
        #
        # print('Saving all collected trajectories')
        # np.save('./data/tmp/trajectories.npy', Ts)

        print('loading saved trajectories')
        Ts = np.load('./data/tmp/trajectories.npy',
                     allow_pickle=True).tolist()

        print('LENGTH OF FIRST TRAJECTORY : ', len(Ts[0]))
        print('AVERAGE LENGTH OF TRAJECTORY : ', [np.average([len(Ts[z]) for z in range(len(Ts))])])

        # Unpack trajectories into features and labels
        X_list = []  # previous states and the action taken at that state
        Y_list = []  # next state
        for T in Ts:  # For each trajectory:
            for i in range(len(T) - self.frames):
                X = np.array([])
                # Append previous self.frames states
                for j in range(self.frames):
                    X = np.append(X, T[i + j][0])
                # Append action
                X = np.append(X, T[i + self.frames - 1][1])
                X_list.append(X)
                Y_list.append(T[i + self.frames][0] - T[i + self.frames - 1][0])

        # store the data so it can be reused in future grounding steps
        self.fwd_model_x_list.extend(X_list)
        self.fwd_model_y_list.extend(Y_list)

        # # normalize the data using mean and standard deviation
        # self.fwd_norms_x = (np.mean(self.fwd_model_x_list, axis=0),
        #                     np.std(self.fwd_model_x_list, axis=0))
        # self.fwd_norms_y = (np.mean(self.fwd_model_y_list, axis=0),
        #                     np.std(self.fwd_model_y_list, axis=0))
        #
        #
        # X_list_normalized = apply_norm(self.fwd_model_x_list, self.fwd_norms_x)
        # Y_list_normalized = apply_norm(self.fwd_model_y_list, self.fwd_norms_y)

        print('STARTING TO TRAIN THE FORWARD MODEL ... ')
        self.fwd_norms_x, self.fwd_norms_y = train_model_es(self.forward_model,
                                                            self.fwd_model_x_list,
                                                            self.fwd_model_y_list,
                                                            self.forward_model_optimizer,
                                                            self.forward_model_criterion)

        print('fwd model norms x : ', self.fwd_norms_x)
        print('fwd model norms y : ', self.fwd_norms_y)

    def train_inverse_model(self, num_traj=NUM_SIM_WORLD_TRAJECTORIES, use_fresh_trajectories=True):
        """Trains the inverse model based on simulated trajectories"""
        print('TRAINING THE INVERSE DYNAMICS MODEL')
        # Trajectories on simulated environment
        if use_fresh_trajectories:
            Ts = collect_gym_trajectories(
                self.sim_env,
                self.target_policy,
                num_traj)

        else:
            print('loading saved trajectories')
            Ts = np.load('./data/tmp/trajectories.npy',
                         allow_pickle=True).tolist()

            # # add some more trajectories
            # Ts.extend(collect_gym_trajectories(self.sim_env,
            #                                    self.target_policy,
            #                                    num_traj))

        print('length of first trajectory : ', len(Ts[0]))

        # Unpack trajectories into features and labels
        X_list = []  # previous states and current state and next state
        Y_list = []  # current action
        for T in Ts:  # For each trajectory:
            for i in range(len(T) - self.frames):
                X = np.array([])

                # Append previous self.frames states
                for j in range(self.frames):
                    X = np.append(X, T[i + j][0])

                # append the delta S (change in state)
                X = np.append(X, T[i + self.frames][0] - T[i + self.frames - 1][0])

                X_list.append(X)
                Y_list.append(T[i + self.frames - 1][1])

        # store the data so it can be reused in future grounding steps
        self.inv_model_x_list.extend(X_list)
        self.inv_model_y_list.extend(Y_list)

        # self.inv_norms_x = (np.mean(self.inv_model_x_list, axis=0),
        #                     np.std(self.inv_model_x_list, axis=0))
        # self.inv_norms_y = (np.mean(self.inv_model_y_list, axis=0),
        #                     np.std(self.inv_model_y_list, axis=0))
        # 
        # 
        # # normalize the data using mean and standard deviation
        # X_list_normalized = apply_norm(self.inv_model_x_list, self.inv_norms_x)
        # Y_list_normalized = apply_norm(self.inv_model_y_list, self.inv_norms_y)

        print('STARTING TO TRAIN THE INVERSE MODEL ... ')
        self.inv_norms_x, self.inv_norms_y = train_model_es(self.inverse_model,
                                                            self.inv_model_x_list,
                                                            self.inv_model_y_list,
                                                            self.inverse_model_optimizer,
                                                            self.inverse_model_criterion)

        print('inv model norms x : ', self.inv_norms_x)
        print('inv model norms y : ', self.inv_norms_y)

    def save_models(self, iter_num=None):
        """Saves forward and inverse models and target policy"""
        iter_path = '' if iter_num is None else '_' + str(iter_num)

        torch.save(self.forward_model.state_dict(),
                   'data/models/forward_model' + iter_path + '.pth')
        torch.save(self.inverse_model.state_dict(),
                   'data/models/inverse_model' + iter_path + '.pth')
        self.target_policy.save('data/models/target_policy' + iter_path + '.pkl')

    def load_models(self,
                    iter_num=None,
                    fwd_load=True,
                    inv_load=True,
                    target_load=True):
        """Loads forward and inverse models and target policy"""
        iter_path = '' if iter_num is None else '_' + str(iter_num)
        if fwd_load:
            self.forward_model.load_state_dict(
                torch.load('data/models/forward_model' + iter_path + '.pth'))
        if inv_load:
            self.inverse_model.load_state_dict(
                torch.load('data/models/inverse_model' + iter_path + '.pth'))
        if target_load:
            self.target_policy.load('data/models/target_policy' + iter_path + '.pkl')
