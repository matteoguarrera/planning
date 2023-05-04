import torch.random
from diffusion_params import get_model_parameters_for_diffusion, get_model_parameters_for_diffusion_from_string
from init_logging import *

from utils.imports import *
from utils.Dataset import load_dataset, show_statistics, normalize_data, unnormalize_data
from utils.Components import ConditionalUnet1D
import argparse

from parse import *
from utils.dynamics import import_dynamics
from utils.dataset_generator import drone_initial_condition


class Diffusion:
    def __init__(self, action_dim,
                 obs_dim,
                 obs_horizon,
                 params,
                 num_training_steps=None,
                 ckpt_path=None,
                 training_flag=True):

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.down_dims = params['DOWN_DIMS']
        self.num_diffusion_iters = params['NUM_DIFFUSION_ITERS']
        self.device = params['DEVICE']
        self.wandb = params['WANDB']

        # create network object
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim * obs_horizon,
            down_dims=self.down_dims,
            diffusion_step_embed_dim=params['DIFFUSION_STEP_EMBEDDING_DIM'],
            kernel_size=params['KERNEL_SIZE'],
            is_m1_arch=params['IS_M1_ARCH']
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=params['NUM_DIFFUSION_ITERS'],
            # the choose of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

        _ = self.noise_pred_net.to(self.device)

        if training_flag:
            # Exponential Moving Average
            # accelerates training and improves stability
            # holds a copy of the model weights
            self.ema = EMAModel(
                model=self.noise_pred_net,
                power=params['EMA_POWER'])

            # Standard ADAM optimizer
            # Note that EMA parameters are not optimized
            if params['OPTIMIZER'] == 'adamw':
                self.optimizer = torch.optim.AdamW(
                    params=self.noise_pred_net.parameters(),
                    lr=params['LEARNING_RATE'], weight_decay=params['WEIGHT_DECAY']
                )
            else:
                raise NotImplementedError("No other optimizer other than AdamW has been tried yet.")

            # Cosine LR schedule with linear warmup
            self.lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=self.optimizer,
                num_warmup_steps=params['COSINE_LR_NUM_WARMUP_STEPS'],
                num_training_steps=num_training_steps
            )

        else:
            # **Loading Pretrained Checkpoint**
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.ema_noise_pred_net = self.noise_pred_net
            self.ema_noise_pred_net.load_state_dict(state_dict)

            print('Pretrained weights loaded.')
            del self.noise_pred_net

    def infer_one_action(self, nobs, pred_horizon):

        B = 1
        # infer action
        with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Gaussian noise
            noisy_action = torch.randn(
                (B, pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.ema_noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        return naction


def inference(obs, diffusion_obj, max_steps, stats, pred_horizon, action_horizon,
            A_mat, B_mat, fn_speed, fn_distance):

    torch.manual_seed(0)  # try to make diffusion model predictable
    OBS = []
    ACTION_PRED = []

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * diffusion_obj.obs_horizon, maxlen=diffusion_obj.obs_horizon)

    rewards = list()
    done = False
    step_idx = 0
    with tqdm(total=max_steps, desc="Eval", leave=False) as pbar:
        while not done:

            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(obs_deque)
            # normalize observation
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            # device transfer
            nobs = torch.from_numpy(nobs).to(diffusion_obj.device, dtype=torch.float32)

            n_action = diffusion_obj.infer_one_action(nobs, pred_horizon)

            # unnormalize action
            # (B, pred_horizon, action_dim)
            n_action = n_action.detach().to('cpu').numpy()[0]

            action_pred = unnormalize_data(n_action, stats=stats['action'])

            # only take action_horizon number of actions
            start = diffusion_obj.obs_horizon - 1  # obs horizon is 2
            end = start + action_horizon
            action = action_pred[start:end, :]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):

                OBS.append(obs)
                ACTION_PRED.append(action)

                # stepping env
                # obs, reward, done, info = env.step(action[i])
                obs = A_mat @ obs + B_mat @ action[i]

                dist = fn_distance(obs)  # how far are we
                speed = fn_speed(obs)

                rew_enable = dist < 1
                reward = rew_enable * (1 - dist)

                done = reward > 0.95 and speed < 0.1

                obs_deque.append(obs)

                rewards.append(reward)

                if done:
                    return (True, reward, speed, max(rewards), step_idx), (OBS, ACTION_PRED)

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                pbar.set_postfix(max_reward=max(rewards))

                if step_idx > max_steps:
                    # too many steps
                    return (False, reward, speed, max(rewards), step_idx), (OBS, ACTION_PRED)


def testing(ckpt_path, max_steps=400, n_sim=100):
    # limit environment interaction to 200 steps before termination
    params = get_model_parameters_for_diffusion_from_string(ckpt_path)

    A_mat, B_mat, x_target, n_state, n_input, _ = import_dynamics(system_name=params['SYSTEM_NAME'])
    dataset, obs_dim, action_dim, _, fn_distance, fn_speed = load_dataset(system_name=params['SYSTEM_NAME'],
                                                                          dtype=params['DTYPE'],
                                                                          device=params['DEVICE'])

    stats = dataset.stats
    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    diffusion_obj = Diffusion(action_dim=action_dim,
                              obs_dim=obs_dim,
                              obs_horizon=obs_horizon,
                              params=params,
                              ckpt_path=ckpt_path,
                              training_flag=False)  # inference

    TRAJECTORIES = []
    REWARD = np.zeros((n_sim, 5))
    for n_sim_idx in range(n_sim):

        # print(n_sim_idx, end=' ')
        np.random.seed(n_sim_idx + 10000)  # seed was between 0 and 500 in the training set

        obs = np.random.uniform(low=-5.0, high=5.0, size=n_state)
        if params['SYSTEM_NAME'] == 'drone':
            obs = drone_initial_condition(obs)
            # obs = zero_velocity_initial_condition(obs)

        REWARD[n_sim_idx, :], trajectory = inference(obs, diffusion_obj, max_steps,
                                                     stats, pred_horizon, action_horizon,
                                                     A_mat, B_mat, fn_speed, fn_distance)
        TRAJECTORIES.append(trajectory)

    return REWARD, TRAJECTORIES, np.sum(REWARD[:, 0])


def training(system='2d'):
    # get hyper-parameters for diffusion model
    params = get_model_parameters_for_diffusion()

    # Import synthetic dataset
    # system_name can be '2d', '3d', 'drone'
    dataset_ours, obs_dim, action_dim, _, fn_distance, fn_speed = load_dataset(system_name=system,
                                                                               dtype=params['DTYPE'],
                                                                               device=params['DEVICE'])

    # dataset_ours, obs_dim, action_dim, name, fn_distance, fn_speed  = load_dataset_lqr2d_observation() #  @carlo

    # Show distribution of trajectories length
    # same dataset as the paper
    show_statistics(dataset_ours=dataset_ours)

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset_ours,
        batch_size=params['BATCH_SIZE'],
        num_workers=params['NUM_WORKERS'],
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=not params['IS_M1_ARCH'],
        # don't kill worker process after each epoch
        persistent_workers=not params['IS_M1_ARCH'],
    )

    TYPE = torch.float32
    obs_horizon = dataset_ours.obs_horizon

    print('Dimension of the hidden layers: ', params['DOWN_DIMS'])

    diffusion_obj = Diffusion(action_dim=action_dim,
                              obs_dim=obs_dim,
                              obs_horizon=obs_horizon,
                              params=params,
                              num_training_steps=len(dataloader) * params['NUM_EPOCHS'])

    # Training Loop
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m_%d_%H_%M_%S")
    num_param = f'{diffusion_obj.noise_pred_net.num_params:.2e}'.replace('+', '').replace('.', '_')

    folder = f"pretrained/{system}_arch{params['ARCH']}_e{params['NUM_EPOCHS']}_d{params['NUM_DIFFUSION_ITERS']}" \
             f"_edim{params['DIFFUSION_STEP_EMBEDDING_DIM']}_ks{params['KERNEL_SIZE']}" \
             f"_par{num_param}_date{date_time}"

    # initialize logging
    if diffusion_obj.wandb:
        init_logging(params, diffusion_obj.noise_pred_net)

    train_loop(dataloader, diffusion_obj, params['NUM_EPOCHS'], system, folder, params['DTYPE'])


def train_loop(dataloader, diffusion_obj, num_epochs, system, folder, dtype):
    print(folder)
    os.makedirs(folder, exist_ok=True)
    writer = SummaryWriter(log_dir=folder)  # log tensorboard

    LOSS = []
    GRADS = []
    with tqdm(range(num_epochs), desc='Epoch', leave=False) as t_global:
        # epoch loop
        for epoch_idx in t_global:
            epoch_loss = list()
            ckpt_filename = f'{system}_{epoch_idx}'

            if epoch_idx % 30 == 0 and epoch_idx > 0:
                torch.save(diffusion_obj.ema.averaged_model.state_dict(), f'./{folder}/model_ema_{ckpt_filename}.ckpt')


            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as t_epoch:
                for n_batch in t_epoch:
                    # data normalized in dataset
                    # device transfer
                    n_obs = n_batch['obs'].to(dtype).to(diffusion_obj.device)
                    n_action = n_batch['action'].to(dtype).to(diffusion_obj.device)

                    B = n_obs.shape[0]

                    # observation as FiLM conditioning
                    # (B, obs_horizon, obs_dim)
                    obs_cond = n_obs[:, :diffusion_obj.obs_horizon, :]
                    # (B, obs_horizon * obs_dim)
                    obs_cond = obs_cond.flatten(start_dim=1).float()

                    # sample noise to add to actions
                    noise = torch.randn(n_action.shape,
                                        device=diffusion_obj.device)  # can be done a priori before starting

                    # sample a diffusion iteration for each data point
                    time_steps = torch.randint(
                        0, diffusion_obj.noise_scheduler.config.num_train_timesteps,
                        (B,), device=diffusion_obj.device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = diffusion_obj.noise_scheduler.add_noise(
                        n_action, noise, time_steps).float()

                    # predict the noise residual
                    # the noise prediction network
                    # takes noisy action, diffusion iteration and observation as input
                    # predicts the noise added to action
                    noise_pred = diffusion_obj.noise_pred_net(
                        sample=noisy_actions, timestep=time_steps, global_cond=obs_cond)

                    # illustration of removing noise
                    # the actual noise removal is performed by NoiseScheduler
                    # and is dependent on the diffusion noise schedule
                    # denoised_action = noise_pred - noise

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    diffusion_obj.optimizer.step()

                    current_grad = {}
                    current_grad = [(name, param.grad.detach().clone().to('cpu'))
                                    for name, param in diffusion_obj.noise_pred_net.named_parameters()]
                    GRADS.append(current_grad)

                    diffusion_obj.optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    diffusion_obj.lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    diffusion_obj.ema.step(diffusion_obj.noise_pred_net)

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    t_epoch.set_postfix(loss=loss_cpu)

            writer.add_scalar('Training/loss_avg', np.mean(epoch_loss), epoch_idx)
            writer.add_scalar('Training/loss_avg', np.std(epoch_loss), epoch_idx)

            if diffusion_obj.wandb:
                wandb.log({'Mean Training Loss': np.mean(epoch_loss)})
                wandb.log({'Training Loss Std': np.std(epoch_loss)})

            LOSS.append(epoch_loss)
            t_global.set_postfix(loss=np.mean(epoch_loss))

    torch.save(diffusion_obj.noise_pred_net.state_dict(), f'./{folder}/model_{ckpt_filename}.ckpt')
    torch.save(diffusion_obj.ema.averaged_model.state_dict(), f'./{folder}/model_ema_{ckpt_filename}.ckpt')

    # fig, ax = plt.subplots(1,1)
    # for l in LOSS[:]:
    #     ax.plot(l)
    # plt.show()

    with open(f'{folder}/LOSS_model_{ckpt_filename}.pickle', 'wb') as handle:
        pickle.dump(LOSS, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Weights of the EMA model
    # is used for inference
    ema_noise_pred_net = diffusion_obj.ema.averaged_model  # 0.000204,    # 0.000517

    return ema_noise_pred_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diffusion Policy For Drone Path Planning')

    parser.add_argument('-test', '--test-mode', default=True, type=bool,
                        help='True if we are testing, False if we are training')

    parser.add_argument('-pm', '--pretrained-model', default='pretrained/2d_arch256_e100_d100_edim256_ks5_'
                                                             'par3_70e06_date04_26_02_22_20/model_ema_2d_99.ckpt',
                        type=str, help='path to pretrained model')

    parser.add_argument('-sn', '--system_name', default='2d', type=str, help='2d or 3d path planning for a drone')
    args = parser.parse_args()
    # download pretrained models and stuff

    os.makedirs('inference', exist_ok=True)
    os.makedirs('pretrained', exist_ok=True)

    print(args)
    test_mode = args.test_mode
    system_name = args.system_name

    if test_mode:
        ckpt_path = args.pretrained_model #'pretrained/2d_arch1024_e100_d50_edim256_ks5_par4_47e07_date04_29_18_53_40/model_ema_2d_30.ckpt'
        testing(ckpt_path, max_steps=400, n_sim=10)  #100
    else:
        training(system=system_name)


    # print(response.json()["name"])

    # parser = argparse.ArgumentParser(description='')
    #
    # parser.add_argument('filename')  # positional argument
    # parser.add_argument('--system_name', default='3d')
    # parser.add_argument('--diffusion_step_embed_dim', default=256)

    # parser.add_argument('--system_name')  # option that takes a value

    # parser.add_argument('-v', '--verbose',
    #                     action='store_true')  # on/off flag
    # args = parser.parse_args()
    # print(args.filename, args.count, args.verbose)

    # ckpt_path = 'pretrained/2d_arch128_e2_d10_edim256_ks3_par1_20e06_date04_25_20_12_22/model_ema_2d_1.ckpt'
    # testing(ckpt_path)

