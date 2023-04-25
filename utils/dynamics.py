import numpy as np


def import_dynamics(system_name: str):
    if system_name == '2d':
        fn_dynamics = __dynamics_2d__

    elif system_name == '3d':
        fn_dynamics = __dynamics_3d__

    elif system_name == 'drone':
        fn_dynamics = __dynamics_drone__

    else:
        raise f'Dynamic system {system_name} not known'

    A_mat, B_mat, x_target, n_state, n_input, name = fn_dynamics()
    return A_mat, B_mat, x_target, n_state, n_input, name


def __dynamics_2d__():
    """ Dynamical system point in 2D """
    Ts = 0.05  # discrete simulation time step

    # dynamic matrices for double integrator system
    A_mat = np.array([[1.0, Ts, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, Ts],
                      [0, 0, 0, 1.0]])
    n_state = 4
    B_mat = np.array([[0, 0],
                      [Ts, 0],
                      [0, 0],
                      [0, Ts]])
    n_input = 2
    # goal state (origin with zero velocity)
    x_target = np.array([[0],
                         [0],
                         [0],
                         [0]])
    name = '2d'
    return A_mat, B_mat, x_target, n_state, n_input, name


def __dynamics_3d__():
    """ Dynamical system point in 3D """
    Ts = 0.05  # discrete simulation time step

    # dynamic matrices for double integrator system
    A_mat = np.array([[1.0, Ts, 0, 0, 0, 0],
                      [0, 1.0, 0, 0, 0, 0],
                      [0, 0, 1.0, Ts, 0, 0],
                      [0, 0, 0, 1.0, 0, 0],
                      [0, 0, 0, 0, 1.0, Ts],
                      [0, 0, 0, 0, 0, 1.0]])
    n_state = 6
    B_mat = np.array([[0, 0, 0],
                      [Ts, 0, 0],
                      [0, 0, 0],
                      [0, Ts, 0],
                      [0, 0, 0],
                      [0, 0, Ts]])
    n_input = 3
    # goal state (origin with zero velocity)
    x_target = np.array([[0],
                         [0],
                         [0],
                         [0],
                         [0],
                         [0]])
    name = '3d'
    return A_mat, B_mat, x_target, n_state, n_input, name


def __dynamics_drone__():
    """ Dynamical system linearized drone """
    Ts = 0.05  # discrete simulation time step

    # dynamic matrices for double integrator system
    g = 9.81
    m = 0.5
    Jx = 0.0039
    Jy = Jx
    Jz = 0.0078
    A_mat = np.array([[1.0, 0, 0, Ts, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1.0, 0, 0, Ts, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1.0, 0, 0, Ts, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1.0, 0, 0, 0, Ts * g, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1.0, 0, -Ts * g, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1.0, 0, 0, Ts, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, Ts, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, Ts],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]])
    n_state = 12
    B_mat = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [1 / m, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0.1 / Jx, 0, 0],
                      [0, 0, 0.1 / Jy, 0],
                      [0, 0, 0, 0.01 / Jz]])
    n_input = 4
    # goal state (origin with zero velocity)
    x_target = np.array([[0],
                         [0],
                         [0],
                         [0],
                         [0],
                         [0],
                         [0],
                         [0],
                         [0],
                         [0],
                         [0],
                         [0]])
    name = 'drone'
    return A_mat, B_mat, x_target, n_state, n_input, name
