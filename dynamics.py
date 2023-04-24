import numpy as np
def dynamics_2D():
    ################ 2D ###################
    Ts = 0.05  # discrete simulation time step

    # dynamic matrices for double integrator system
    A_mat = np.array([[1.0, Ts, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, Ts],
                  [0, 0, 0, 1.0]])
    n_state = 4
    B_mat = np.array([[0,0],
                  [Ts,0],
                  [0,0],
                  [0,Ts]])
    n_input = 2
    # goal state (origin with zero velocity)
    x_target = np.array([[0],
                         [0],
                         [0],
                         [0]])
    name = 'LQR2D'
    return A_mat, B_mat, x_target, n_state, n_input, name

def dynamics_3D():
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
    name = 'LQR3D'
    return A_mat, B_mat, x_target, n_state, n_input, name


def dynamics_Drone():
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
    name = 'LQRDrone'
    return A_mat, B_mat, x_target, n_state, n_input, name