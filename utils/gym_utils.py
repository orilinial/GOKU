import numpy as np
import skimage.transform


def get_theta(obs):
    """Transforms coordinate basis from the defaults of the gym pendulum env."""
    theta = np.arctan2(obs[0], -obs[1])
    theta = theta + np.pi / 2
    theta = theta + 2 * np.pi if theta < -np.pi else theta
    theta = theta - 2 * np.pi if theta > np.pi else theta
    return theta


def preproc(X, side):
    """Crops, downsamples, desaturates, etc. the rgb pendulum observation."""
    X = X[..., 0][440:-220, 330:-330] - X[..., 1][440:-220, 330:-330]
    return skimage.transform.resize(X, [int(side), side]) / 255.


def step_env(args, env, u, params, additional_params):
    th, thdot = env.state  # th := theta

    g = 10.0
    m = 1.0
    b = additional_params['b']
    l = params["l"]
    dt = env.dt

    # u = np.clip(u, -env.max_torque, env.max_torque)[0]
    # env.last_u = u # for rendering
    # costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

    # newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt

    if args.friction:
        newthdot = thdot + ((-g / l) * np.sin(th + np.pi) - (b / m) * thdot) * dt
    else:
        newthdot = thdot + ((-g / l) * np.sin(th + np.pi)) * dt

    newth = th + newthdot * dt
    newthdot = np.clip(newthdot, -env.max_speed, env.max_speed)  # pylint: disable=E1111

    env.state = np.array([newth, newthdot])
    return env._get_obs()  # , -costs, False, {}


def get_params():
    l = 1.0
    # l = np.random.uniform(1.0, 2.0)
    # l = 1.0 if np.random.rand() > 0.5 else 2.0
    params = {"l": l}
    return params


def get_unlearned_params():
    b = 1.0
    # l = np.random.uniform(1.0, 2.0)
    # l = 1.0 if np.random.rand() > 0.5 else 2.0
    params = {"b": b}
    return params


def reset_env(env, min_angle=0., max_angle=np.pi / 6):
    angle_ok = False
    while not angle_ok:
        obs = env.reset()
        theta_init = np.abs(get_theta(obs))
        if min_angle < theta_init < max_angle:
            angle_ok = True
