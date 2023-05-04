import os
import numpy as np
import scipy
import pickle

from utils.dynamics import import_dynamics


def drone_initial_condition(obs0):
    obs0[6:8] = np.random.uniform(low=-1.0, high=1.0, size=(2, 1))
    obs0[8] = np.random.uniform(low=0, high=2 * np.pi)
    obs0[9:12] = np.random.uniform(low=-1.0, high=1.0, size=(3, 1))
    return obs0


def zero_velocity_initial_condition(obs0):
    obs_dim = obs0.shape[1]

    if obs_dim == 4:
        obs0[1], obs0[3] = 0, 0  # 2d
    if obs_dim == 6:
        obs0[1], obs0[3], obs0[5] = 0, 0, 0  # 3d
    if obs_dim > 8:
        obs0[3:] = 0  # 9, 10, 11

    raise NotImplementedError

    return obs0


def generate_dataset(system_name, baseline=False):
    """ generate a dataset using LQR algorithm, please refer to the paper for further details
        system name: ['2d', '3d', 'drone']
    """
    print(f'[Dataset] Generating Dataset {system_name}')
    A, B, x_target, n_state, n_input, name = import_dynamics(system_name=system_name)

    n_rand_controllers = 500  # how many random controllers we want to generate
    rollout_len = 250  # how many time steps a rollout is
    finish_threshold = 0.02  # stopping condition

    x_data = np.zeros((n_rand_controllers * rollout_len, n_state))
    u_data = np.zeros((n_rand_controllers * rollout_len, n_input))
    finish_token = np.zeros((n_rand_controllers * rollout_len,))
    if baseline:
        stats = {'rollout_len': np.zeros(n_rand_controllers),  # max length of the rollout
                 'i_rollout': np.zeros(n_rand_controllers),  # length of the actual rollout
                 'dist_to_target': np.zeros(n_rand_controllers),  # last dist to target
                 'x0': np.zeros((n_rand_controllers, n_state))}

    for i_contr in range(n_rand_controllers):
        if baseline:
            np.random.seed(i_contr + 10000)
        else:
            np.random.seed(i_contr)

        x0 = np.random.uniform(low=-5.0, high=5.0, size=(n_state, 1))
        if name == 'drone':
            x0 = drone_initial_condition(x0)

        x_data[i_contr * rollout_len, :] = x0.reshape(n_state, )

        Q = np.random.uniform(low=0.0, high=1.0, size=(n_state, n_state))
        Q = Q @ Q.T

        R = np.random.uniform(low=0.0, high=1.0, size=(n_input, n_input))
        R = R @ R.T

        # steady state lqr
        X = scipy.linalg.solve_discrete_are(A, B, Q, R)
        F = np.linalg.inv(R + B.T @ X @ B) @ B.T @ X @ A

        for i_rollout in range(rollout_len):
            dist_to_target = np.linalg.norm(x_data[i_contr * rollout_len + i_rollout, :] - x_target)
            if dist_to_target < finish_threshold:
                finish_token[i_contr * rollout_len + i_rollout] = 1.0
                break
            if i_rollout == (rollout_len - 1):
                finish_token[i_contr * rollout_len + i_rollout] = 1.0
                break
            u_data[i_contr * rollout_len + i_rollout, :] = -F @ x_data[i_contr * rollout_len + i_rollout, :]
            x_data[i_contr * rollout_len + i_rollout + 1, :] = A @ x_data[i_contr * rollout_len + i_rollout, :] + \
                                                               B @ u_data[i_contr * rollout_len + i_rollout, :]

        if baseline:
            stats['rollout_len'][i_contr] = rollout_len  # max length of the rollout
            stats['i_rollout'][i_contr] = i_rollout  # length of the actual rollout
            stats['dist_to_target'][i_contr] = dist_to_target  # last dist to target
            stats['x0'][i_contr] = x0[:, 0]  # initial condition

    aux = (u_data == np.zeros((1, n_input))).sum(axis=1)
    aux1 = np.setdiff1d(np.nonzero(aux), np.nonzero(finish_token))
    x_data = np.delete(x_data, aux1, 0)
    u_data = np.delete(u_data, aux1, 0)
    finish_token = np.delete(finish_token, aux1, 0)

    data = {'obs': x_data,
            'action': u_data,
            'finish_token': finish_token}

    os.makedirs('datasets', exist_ok=True)
    with open(f'datasets/dataset_{name}.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    if baseline:
        with open(f'datasets/dataset_{name}_baseline_stats.pkl', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

        with open(f'datasets/dataset_{name}_baseline_data.pkl', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

