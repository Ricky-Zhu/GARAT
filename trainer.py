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

warnings.simplefilter(action='ignore', category=FutureWarning)
from rl_gat.reinforcedgat import ReinforcedGAT
import gym, os, glob, shutil
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2, TRPO
import argparse, sys
from termcolor import cprint
from scripts.utils import MujocoNormalized
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
import torch
import random
from rl_gat.envs import *
from tqdm import tqdm
from datetime import datetime

torch.backends.cudnn.deterministic = True


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


def check_tmp_folder():
    """
    function to check if the tmp folder is clean, for storing
    trajectories as numpy files in tmp folder.
    :return:python
    """
    if os.path.exists('./data/tmp'):
        print('data/tmp/ directory already exists. Deleting all files. ')
        shutil.rmtree('./data/tmp')

    try:
        os.mkdir('./data/tmp')
        print('Successfully created tmp folder. ')

    except Exception as e:
        print(e)


def evaluate_policy_on_env(env,
                           model,
                           render=True,
                           iters=1,
                           deterministic=True,
                           save_the_optim_traj_states=False
                           ):
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
        return_list.append(return_val)

    # sample 2000 data points to represent state visitation
    if save_the_optim_traj_states:
        state_visit_to_save = random.sample(state_visit, k=2000)
    if save_the_optim_traj_states:
        return np.mean(return_list), np.std(return_list) / np.sqrt(len(return_list)), state_visit_to_save
    else:
        return np.mean(return_list), np.std(return_list) / np.sqrt(len(return_list))


def main():
    parser = argparse.ArgumentParser(description='Reinforced Grounded Action Transformation')
    parser.add_argument('--target_policy_algo', default="TRPO", type=str,
                        help="name in str of the agent policy training algorithm")
    parser.add_argument('--action_tf_policy_algo', default="PPO2", type=str,
                        help="name in str of the Action Transformer policy training algorithm")
    parser.add_argument('--load_policy_path',
                        default='data/models/TRPO_initial_policy_steps_HalfCheetah-v2_2500000_.pkl',
                        help="relative path of initial policy trained in sim")
    parser.add_argument('--alpha', default=1.0, type=float, help="Deprecated feature. Ignore")
    parser.add_argument('--beta', default=1.0, type=float, help="the reward scale set in ATP env")
    parser.add_argument('--n_trainsteps_target_policy', default=1000, type=int,
                        help="Number of time steps to train the agent policy in the grounded environment")
    parser.add_argument('--n_trainsteps_action_tf_policy', default=1000000, type=int,
                        help="Timesteps to train the Action Transformer policy in the ATPEnvironment")
    parser.add_argument('--num_cores', default=10, type=int,
                        help="Number of threads to use while collecting real world experience")
    parser.add_argument('--sim_env', default='HalfCheetah-v2',
                        help="Name of the simulator environment (Unmodified)")
    parser.add_argument('--real_env', default='HalfCheetahModified-v2',
                        help="Name of the Real World environment (Modified)")
    parser.add_argument('--n_frames', default=1, type=int, help="Number of previous frames observed by discriminator")
    parser.add_argument('--expt_number', default=1, type=int, help="Expt. number to keep track of multiple experiments")
    parser.add_argument('--n_grounding_steps', default=1, type=int,
                        help="Number of grounding steps. (Outerloop of algorithm ) ")
    parser.add_argument('--n_iters_atp', default=1, type=int, help="Number of GAN iterations")
    parser.add_argument('--discriminator_epochs', default=1, type=int, help="Discriminator epochs per GAN iteration")
    parser.add_argument('--generator_epochs', default=50, type=int, help="ATP epochs per GAN iteration")
    parser.add_argument('--real_trajs', default=100, type=int, help="Set max amount of real TRAJECTORIES used")
    parser.add_argument('--sim_trajs', default=100, type=int, help="Set max amount of sim TRAJECTORIES used")
    parser.add_argument('--real_trans', default=50, type=int, help="amount of real world transitions used")
    parser.add_argument('--gsim_trans', default=50, type=int, help="amount of simulator transitions used")
    parser.add_argument('--eval', action='store_true',
                        help="set to true to evaluate the agent policy in the real environment, after training in grounded environment")
    parser.add_argument('--use_cuda', action='store_true', help="DEPRECATED. Not using CUDA")
    parser.add_argument('--instance_noise', action='store_true', help="DEPRECATED. Not using instance noise")
    parser.add_argument('--ent_coeff', default=0.00005, type=float,
                        help="entropy coefficient for the PPO algorithm, used to train the action transformer policy")
    parser.add_argument('--max_kl', default=0.000005, type=float,
                        help="Set this only if using TRPO for the action transformer policy")
    parser.add_argument('--clip_range', default=0.1, type=float,
                        help="PPO objective clipping factor -> Action transformer policy")
    parser.add_argument('--plot', action='store_true',
                        help="visualize the action transformer policy - works well only for simple environments")
    parser.add_argument('--tensorboard', action='store_true', help="visualize training in tensorboard")
    parser.add_argument('--save_atp', action='store_true', help="Saves the action transformer policy")
    parser.add_argument('--save_target_policy', action='store_true', help="saves the agent policy")
    parser.add_argument('--loss_function', default="GAIL", type=str,
                        help="choose from the list: ['GAIL', 'WGAN', 'AIRL', 'FAIRL']")
    parser.add_argument('--namespace', default="wed_night", type=str, help="namespace for the experiments")
    parser.add_argument('--dont_reset', action='store_true', help="UNUSED")
    parser.add_argument('--reset_target_policy', action='store_true', help="UNUSED")
    parser.add_argument('--randomize_target_policy', action='store_true', help="UNUSED")
    parser.add_argument('--compute_grad_penalty', action='store_true',
                        help="set this to true to compute the GP term while training the discriminator")
    parser.add_argument('--single_batch_test', action='store_true',
                        help="performs a single update of the generator and discriminator.")
    parser.add_argument('--folder_namespace', default="None", type=str, help="UNUSED")
    parser.add_argument('--disc_lr', default=3e-3, type=float,
                        help="learning rate for the AdamW optimizer to update the discriminator")
    parser.add_argument('--atp_lr', default=3e-4, type=float,
                        help="learning rate for the Adam optimizer to update the agent policy")
    parser.add_argument('--nminibatches', default=4, type=int,
                        help="Number of minibatches used by the PPO algorithm to update the action transformer policy")
    parser.add_argument('--noptepochs', default=4, type=int,
                        help="Number of optimization epochs performed per minibatch by the PPO algorithm to update the action transformer policy")
    parser.add_argument('--deterministic', default=0, type=int,
                        help="set to 0 to use the deterministic action transformer policy in the grounded environment")
    parser.add_argument('--single_batch_size', default=256, type=int, help="batch size for the GARAT update")
    parser.add_argument('--deterministic_sample_collecting', action='store_true',
                        help="deterministically collect samples")

    args = parser.parse_args()

    # set the seeds here for experiments
    random.seed(args.expt_number)
    np.random.seed(args.expt_number)
    torch.manual_seed(args.expt_number)

    # make dummy gym environment
    dummy_env = gym.make(args.real_env)

    current_date = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    expt_label = args.namespace + '_real_' + str(args.real_env) + '_sim_' + str(args.sim_env) + '-' + current_date
    # create the experiment folder
    expt_path = '../data/models/garat/' + expt_label
    expt_already_running = False

    gatworld = ReinforcedGAT(
        load_policy=args.load_policy_path,  # the path of the pretrained policy in the source env
        num_cores=args.num_cores,
        sim_env_name=args.sim_env,
        real_env_name=args.real_env,
        expt_label=expt_label,
        frames=args.n_frames,
        algo=args.target_policy_algo,  # agent policy: trpo
        atp_algo=args.action_tf_policy_algo,  # action transformer policy: ppo
        use_cuda=args.use_cuda,
        real_trans=args.real_trans,  # the collected transition limit once
        gsim_trans=args.gsim_trans,
        expt_path=expt_path,
        tensorboard=args.tensorboard,
        atp_loss_function=args.loss_function,  # here GAIL is used
        single_batch_size=None if args.single_batch_size == 0 else args.single_batch_size,  # TODO: check this usage
        deterministic_sample_collecting=args.deterministic_sample_collecting,
        beta=args.beta

    )

    # first see the performance descrepency of load policy in the source env and the target env
    cprint('first check if there are performance descrepency', 'red', 'on_green')
    val_first_sim = evaluate_policy_on_env(gym.make(args.sim_env),
                                           gatworld.target_policy,
                                           render=False,
                                           iters=20,
                                           deterministic=True)
    val_first_target = evaluate_policy_on_env(gym.make(args.real_env),
                                              gatworld.target_policy,
                                              render=False,
                                              iters=20,
                                              deterministic=True)
    cprint('the initial agent policy performance {} on source env, {} on target env'.format(val_first_sim,
                                                                                            val_first_target), 'green')

    print('running experiment')
    os.makedirs(expt_path)
    grounding_step = 0

    try:  # save the argments TODO: change to args save instead of command line save
        with open(expt_path + '/commandline_args.txt', 'w') as f:
            f.write('\n'.join(sys.argv[1:]))

    except:
        pass

    start_grounding_step = grounding_step

    for _ in range(args.n_grounding_steps - start_grounding_step):

        grounding_step += 1
        cprint('grounding step {}/{}'.format(grounding_step, args.n_grounding_steps), 'blue')

        gatworld.collect_experience_from_real_env()

        cprint('~~ RESETTING DISCRIMINATOR AND ATP POLICY ~~', 'yellow')
        # create the discriminator and the ATP and its training env
        gatworld._init_rgat_models(algo=args.action_tf_policy_algo,
                                   ent_coeff=args.ent_coeff,
                                   # these policy parameters are for the action transformation policy
                                   max_kl=args.max_kl,
                                   clip_range=args.clip_range,
                                   atp_loss_function=args.loss_function,
                                   disc_lr=args.disc_lr,
                                   atp_lr=args.atp_lr,
                                   nminibatches=args.nminibatches,
                                   noptepochs=args.noptepochs,
                                   )

        # ground the environment
        print('enter grounding inner loop for grounding step {}....'.format(grounding_step))
        for ii in tqdm(range(args.n_iters_atp)):
            for _ in range(args.discriminator_epochs):
                gatworld.train_discriminator(iter_step=ii,
                                             grounding_step=grounding_step,
                                             num_epochs=args.noptepochs * 5 if ii <= 10 else args.noptepochs,  # warmup
                                             compute_grad_penalty=args.compute_grad_penalty,
                                             nminibatches=args.nminibatches,
                                             single_batch_test=args.single_batch_test,
                                             )

            gatworld.train_action_transformer_policy(
                                                     num_epochs=args.generator_epochs,
                                                     loss_function=args.loss_function,
                                                     single_batch_test=args.single_batch_test,
                                                     )

            # test grounded environment
            if args.plot and dummy_env.action_space.shape[0] < 5:
                # action transformer plot
                gatworld.test_grounded_environment(alpha=args.alpha,
                                                   grounding_step=str(grounding_step) + '_' + str(ii),
                                                   )
            else:
                # print('Environment has action space > 5. Skipping AT plotting')
                pass

            if args.save_atp:
                # save the action transformer policy for further analysis
                gatworld.save_atp(grounding_step=str(grounding_step) + '_' + str(ii))

        gatworld.train_target_policy_in_grounded_env(grounding_step=grounding_step,
                                                     alpha=args.alpha,
                                                     time_steps=args.n_trainsteps_target_policy,
                                                     save_model=args.save_target_policy,
                                                     use_deterministic=True if args.deterministic == 1 else False,
                                                     )

        if args.eval:
            cprint('Evaluating target policy in environment for grounding step {}'.format(grounding_step), 'red',
                   'on_blue')
            test_env = gym.make(args.real_env)
            test_sim_env = gym.make(args.sim_env)
            if 'mujoco_norm' in args.load_policy_path:
                test_env = MujocoNormalized(test_env)
            elif 'normalized' in args.load_policy_path:
                test_env = DummyVecEnv([lambda: test_env])
                test_env = VecNormalize.load('data/models/env_stats/' + args.sim_env + '.pkl',
                                             venv=test_env)
            # evaluate on the real world.
            try:
                val_sim = evaluate_policy_on_env(test_sim_env,
                                                 gatworld.target_policy,
                                                 render=False,
                                                 iters=20,
                                                 deterministic=True)

                # evaluate the target agent policy in the target env determinsticly and stochasticly

                *val_det, state_to_save = evaluate_policy_on_env(test_env,
                                                                 gatworld.target_policy,
                                                                 render=False,
                                                                 iters=20,
                                                                 deterministic=True,
                                                                 save_the_optim_traj_states=True)

                # save the state
                state_to_save = np.asarray(state_to_save)
                np.save(expt_path + '/state_visit_grounding_step_{}.npy'.format(grounding_step), state_to_save)

                val_stochastic = evaluate_policy_on_env(test_env,
                                                        gatworld.target_policy,
                                                        render=False,
                                                        iters=20,
                                                        deterministic=False)
                print(
                    'grounding step : {}, sim_env_det:{},real_env_det:{},real_env_stochastic:{}'.format(grounding_step,
                                                                                                        val_sim,
                                                                                                        val_det,
                                                                                                        val_stochastic))
                # with open(expt_path + "/stochastic_output.txt", "a") as txt_file:
                #     print(val, file=txt_file)
            except Exception as e:
                cprint(e, 'red')

    os._exit(0)


if __name__ == '__main__':
    main()
    os._exit(0)
