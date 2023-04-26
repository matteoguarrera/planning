from utils.imports import *
from utils.Dataset import load_dataset, show_statistics
from utils.Components import ConditionalUnet1D
import argparse


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

    folder = f'pretrained/{system_name}_arch{arch}_e{num_epochs}_edim{diffusion_step_embed_dim}' \
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
    shrink = 1  # how much small the network wrt papers
    down_dims = [1024 // shrink] #256 // shrink, 512 // shrink,

    training(system_name='2d',
             diffusion_step_embed_dim=256,
             kernel_size=5,
             down_dims=down_dims,
             num_epochs=100,
             num_diffusion_iters=50)
