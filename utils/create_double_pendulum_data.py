import numpy as np
import gym
import skimage.transform
from tqdm import trange


def preproc(X, side):
    """Crops, downsamples, desaturates, etc. the rgb pendulum observation."""
    X = X[..., 0][440:, 150:-150] - X[..., 1][440:, 150:-150]
    return skimage.transform.resize(X, [int(side), side]) / 255.


def dsdt(s, t, args):
    params = args
    m1 = 1.0
    m2 = params["m2"]
    l1 = 1.0
    lc1 = 0.5
    lc2 = 0.5
    I1 = 1.0
    I2 = 1.0
    g = 9.8

    a = 0.0

    theta1 = s[0]
    theta2 = s[1]
    dtheta1 = s[2]
    dtheta2 = s[3]
    d1 = m1 * lc1 ** 2 + m2 * \
         (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
    phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
           - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2) \
           + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
    ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
               / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return (dtheta1, dtheta2, ddtheta1, ddtheta2)


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi, ti)``
        y0: initial state vector
        t: sample times
        args: additional arguments passed to the derivative function
        kwargs: additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    return yout


def step_env(args, env, params):
    s = env.state
    ns = rk4(dsdt, s, [0, env.dt], args=params)
    ns = ns[-1]
    env.state = ns
    return ns


def get_params():
    m2 = np.random.uniform(1.0, 2.0)
    params = {"m2": m2}
    return params


def reset_env(env, min_angle=np.pi/10, max_angle=np.pi/6):
    state = np.random.uniform(low=min_angle, high=max_angle, size=(4,))
    env.state = state
    return state


def create_double_pendulum_data(args, side=32):
    env_name = 'Acrobot-v1'
    env = gym.make(env_name).unwrapped
    env.dt = args.delta_t
    env.seed(args.seed)
    data = np.zeros((args.data_size, args.seq_len, side, side))
    latent_data = np.zeros((args.data_size, args.seq_len, 4))
    params_data = []

    for trial in trange(args.data_size):
        reset_env(env)
        params = get_params()

        for step in range(args.seq_len):
            processed_frame = preproc(env.render('rgb_array'), side)

            data[trial, step] = processed_frame
            obs = step_env(args, env, params)
            latent_data[trial, step] = obs

        params_data.append(params)
    env.close()
    return data, latent_data, params_data
