import os

import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt

from dynamics import *

# A, B, x_target, n_state, n_input, name = dynamics_2D()
# A, B, x_target, n_state, n_input, name = dynamics_3D()
A, B, x_target, n_state, n_input, name = dynamics_Drone()

n_rand_controllers = 500  # how many random controllers we want to generate
rollout_len = 250  # how many time steps a rollout is

x_data = np.zeros((n_rand_controllers * rollout_len, n_state))
u_data = np.zeros((n_rand_controllers * rollout_len, n_input))
finish_token = np.zeros((n_rand_controllers * rollout_len,))

for i_contr in range(n_rand_controllers):

    np.random.seed(i_contr)
    Q = np.random.uniform(low=0.0, high=1.0, size=(n_state, n_state))
    Q = Q @ Q.T  # + np.eye(n_state)
    #     Q[0,0] += 10
    #     Q[2,2] += 10
    R = np.random.uniform(low=0.0, high=1.0, size=(n_input, n_input))
    R = R @ R.T

    # steady state lqr
    X = scipy.linalg.solve_discrete_are(A, B, Q, R)
    F = np.linalg.inv(R + B.T @ X @ B) @ B.T @ X @ A
    #     print(F.shape)

    x0 = np.random.uniform(low=-5.0, high=5.0, size=(n_state, 1))
    x_data[i_contr * rollout_len, :] = x0.reshape(n_state, )

    for i_rollout in range(rollout_len):

        if np.linalg.norm(x_data[i_contr * rollout_len + i_rollout, :] - x_target) < 0.05:
            #             print('finish1')
            finish_token[i_contr * rollout_len + i_rollout] = 1.0
            break
        if i_rollout == (rollout_len - 1):
            #             print('finish2')
            finish_token[i_contr * rollout_len + i_rollout] = 1.0
            break
        u_data[i_contr * rollout_len + i_rollout, :] = -F @ x_data[i_contr * rollout_len + i_rollout, :]
        x_data[i_contr * rollout_len + i_rollout + 1, :] = A @ x_data[i_contr * rollout_len + i_rollout,
                                                               :] + B @ u_data[i_contr * rollout_len + i_rollout, :]

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