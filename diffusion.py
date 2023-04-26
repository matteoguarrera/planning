from utils.imports import *
from utils.Dataset import load_dataset, show_statistics
from utils.Components import ConditionalUnet1D
import argparse

from parse import *
from utils.dynamics import import_dynamics
from utils.Dataset import normalize_data, unnormalize_data, load_dataset


def testing(ckpt_path, max_steps= 400, n_sim= 100):
    # limit enviornment interaction to 200 steps before termination

    result = parse('pretrained/{}_arch{}_e{}_d{}_edim{}_ks{}_par{}_date{}', ckpt_path)

    system_name, arch, num_epochs, num_diffusion_iters, diffusion_step_embed_dim, kernel_size, num_param, date_time = result
    num_epochs = int(num_epochs)
    num_diffusion_iters = int(num_diffusion_iters)
    diffusion_step_embed_dim = int(diffusion_step_embed_dim)
    kernel_size = int(kernel_size)
    down_dims = [int(i) for i in arch.split('_')]

    A_mat, B_mat, x_target, n_state, n_input, _ = import_dynamics(system_name)
    dataset, obs_dim, action_dim, _, fn_distance, fn_speed = load_dataset(system_name)

    stats = dataset.stats

    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        down_dims=down_dims,
        diffusion_step_embed_dim=diffusion_step_embed_dim,  # 256
        kernel_size=kernel_size,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # device transfer
    device = torch.device('cuda')
    _ = noise_pred_net.to(device)

    # @markdown ### **Loading Pretrained Checkpoint**
    state_dict = torch.load(ckpt_path, map_location='cuda')
    ema_noise_pred_net = noise_pred_net
    ema_noise_pred_net.load_state_dict(state_dict)
    print('Pretrained weights loaded.')

    #####################################################################################

    ended_correctly = 0
    REWARD = np.zeros((n_sim, 4))

    for n_sim_idx in range(n_sim):
        print(n_sim_idx, end=' ')
        np.random.seed(n_sim_idx + 10000)  # seed was between 0 and 500 in the training set

        obs = np.random.uniform(low=-5.0, high=5.0, size=(n_state))
        # obs = np.random.random(obs_dim) * 10 - 5
        if system_name == 'drone':
            obs[6:8] = np.random.uniform(low=-1.0, high=1.0, size=(2, 1))
            obs[8] = np.random.uniform(low=0, high=2 * np.pi)
            obs[9:12] = np.random.uniform(low=-1.0, high=1.0, size=(3, 1))

        # WE SHOULD STRUCTURE THIS BETTER get first observation

        # obs = np.array([ 0.81204461, -2.81407099, -2.26550367, -1.43074666])
        # if obs_dim == 4:
        #     obs[1], obs[3] = 0, 0 # 2d
        # if obs_dim == 6:
        #     obs[1], obs[3], obs[5] = 0, 0, 0 # 3d
        # if obs_dim > 8:
        #     obs[3:] = 0 # 9, 10, 11,

        OBS = []
        ACTION_PRED = []
        ############ obs
        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        # save visualization and rewards
        # imgs = [env.render(mode='rgb_array')]
        rewards = list()
        done = False
        step_idx = 0
        with tqdm(total=max_steps, desc="Eval", leave=False) as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon (2) number of observations
                obs_seq = np.stack(obs_deque)
                # normalize observation
                nobs = normalize_data(obs_seq, stats=stats['obs'])
                # device transfer
                nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

                # infer action
                with torch.no_grad():
                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                    # initialize action from Gaussian noise
                    noisy_action = torch.randn(
                        (B, pred_horizon, action_dim), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    for k in noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = ema_noise_pred_net(
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )

                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                # unnormalize action
                naction = naction.detach().to('cpu').numpy()

                # print('naction: ', action_pred.shape)
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=stats['action'])

                # print('action_pred: ', action_pred.shape)

                # only take action_horizon number of actions
                start = obs_horizon - 1  # obs horizon is 2
                end = start + action_horizon
                action = action_pred[start:end, :]
                # (action_horizon, action_dim)
                # print('action_pred: ', action.shape)

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

                    # and reward/vis
                    rewards.append(reward)
                    # imgs.append(env.render(mode='rgb_array'))

                    if done:
                        ended_correctly += 1
                        REWARD[n_sim_idx, :] = reward, speed, max(rewards), step_idx

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True

                    if done:
                        break

        # print out the maximum target coverage
        print('Score: ', max(rewards))

        # visualize
        # from IPython.display import Video
        # vwrite('vis.mp4', imgs)
        # Video('vis.mp4', embed=True, width=256, height=256)

        return REWARD


def training(system_name='2d',
             diffusion_step_embed_dim=256,
             kernel_size=5,
             down_dims=[256, 512, 1024],
             num_epochs=100,
             num_diffusion_iters=100):

    # Import synthetic dataset
    # system_name can be '2d', '3d', 'drone'
    dataset_ours, obs_dim, action_dim, name, fn_distance, fn_speed = load_dataset(system_name=system_name)

    # dataset_ours, obs_dim, action_dim, name, fn_distance, fn_speed  = load_dataset_lqr2d_observation() # to clean,  @carlo

    # Show distribution of trajectories length
    # same dataset as the paper
    show_statistics(dataset_ours=dataset_ours)

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset_ours,
        batch_size=256,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True
    )

    # # visualize data in batch
    # batch = next(iter(dataloader))
    # print("batch['obs'].shape:", batch['obs'].shape)
    # print("batch['action'].shape", batch['action'].shape)

    TYPE = torch.float32
    obs_horizon = dataset_ours.obs_horizon
    arch = str(down_dims)[1:-1].replace(', ', '_')
    print('Dimension of the hidden layers: ', down_dims)

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        down_dims=down_dims,
        diffusion_step_embed_dim=diffusion_step_embed_dim,  # haven't tuned yet, 256
        kernel_size=kernel_size,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # device transfer
    device = torch.device('cuda')
    _ = noise_pred_net.to(device)

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        model=noise_pred_net,
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    # Training Loop
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m_%d_%H_%M_%S")
    num_param = f'{noise_pred_net.num_params:.2e}'.replace('+', '').replace('.', '_')

    folder = f'pretrained/{system_name}_arch{arch}_e{num_epochs}_d{num_diffusion_iters}_edim{diffusion_step_embed_dim}' \
             f'_ks{kernel_size}_par{num_param}_date{date_time}'

    train_loop(dataloader,
               noise_pred_net, ema, optimizer, lr_scheduler, noise_scheduler,
               num_epochs, device,
               system_name, folder)


def train_loop(dataloader,
               noise_pred_net,
               ema,
               optimizer,
               lr_scheduler,
               noise_scheduler,
               num_epochs,
               device,
               system_name,
               folder):
    print(folder)
    os.makedirs(folder, exist_ok=True)

    obs_horizon = dataloader.dataset.obs_horizon

    writer = SummaryWriter(log_dir=folder)  # log tensorboard

    LOSS = []
    with tqdm(range(num_epochs), desc='Epoch', leave=False) as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            ckpt_filename = f'{system_name}_{epoch_idx}'

            if epoch_idx % 30 == 0:
                torch.save(ema.averaged_model.state_dict(), f'./{folder}/model_ema_{ckpt_filename}.ckpt')

            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)
                    B = nobs.shape[0]

                    # observation as FiLM conditioning
                    # (B, obs_horizon, obs_dim)
                    obs_cond = nobs[:, :obs_horizon, :]
                    # (B, obs_horizon * obs_dim)
                    obs_cond = obs_cond.flatten(start_dim=1).float()

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)  # can be done a priori before starting

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps).float()

                    # predict the noise residual
                    # the noise prediction network
                    # takes noisy action, diffusion iteration and observation as input
                    # predicts the noise added to action
                    noise_pred = noise_pred_net(
                        sample=noisy_actions, timestep=timesteps, global_cond=obs_cond)

                    # illustration of removing noise
                    # the actual noise removal is performed by NoiseScheduler
                    # and is dependent on the diffusion noise schedule
                    denoised_action = noise_pred - noise

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(noise_pred_net)

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            writer.add_scalar('Training/loss_avg', np.mean(epoch_loss), epoch_idx)
            writer.add_scalar('Training/loss_std', np.std(epoch_loss), epoch_idx)

            LOSS.append(epoch_loss)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
    torch.save(noise_pred_net.state_dict(), f'./{folder}/model_{ckpt_filename}.ckpt')
    torch.save(ema.averaged_model.state_dict(), f'./{folder}/model_ema_{ckpt_filename}.ckpt')

    # Weights of the EMA model
    # is used for inference
    ema_noise_pred_net = ema.averaged_model  # 0.000204,    # 0.000517

    # fig, ax = plt.subplots(1,1)
    # for l in LOSS[:]:
    #     ax.plot(l)
    # plt.show()

    with open(f'{folder}/LOSS_model_{ckpt_filename}.pickle', 'wb') as handle:
        pickle.dump(LOSS, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
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

    ckpt_path = 'pretrained/2d_arch128_e2_d10_edim256_ks3_par1_20e06_date04_25_20_12_22/model_ema_2d_1.ckpt'
    testing(ckpt_path)

    shrink = 1  # how much small the network wrt papers
    down_dims = [1024 // shrink] #256 // shrink, 512 // shrink,

    training(system_name='2d',
             diffusion_step_embed_dim=256,
             kernel_size=5,
             down_dims=down_dims,
             num_epochs=100,
             num_diffusion_iters=50)
