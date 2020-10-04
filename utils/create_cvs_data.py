import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import torch


def init_random_state():
    max_ves = 64.0 - 10.0
    min_ves = 36.0 + 10.0

    max_ved = 167.0 - 10.0
    min_ved = 121.0 + 10.0

    max_sv = 1.0
    min_sv = 0.9

    max_pa = 85.0
    min_pa = 75.0

    max_pv = 7.0
    min_pv = 3.0

    max_s = 0.25
    min_s = 0.15

    init_ves = (np.random.rand() * (max_ves - min_ves) + min_ves) / 100.0
    # init_ves = 50.0 / 100.0

    init_ved = (np.random.rand() * (max_ved - min_ved) + min_ved) / 100.0
    # init_ved = 144.0 / 100.0

    init_sv = (np.random.rand() * (max_sv - min_sv) + min_sv)
    init_pa = (np.random.rand() * (max_pa - min_pa) + min_pa) / 100.0
    init_pv = (np.random.rand() * (max_pv - min_pv) + min_pv) / 10.0
    init_s = (np.random.rand() * (max_s - min_s) + min_s)

    init_state = np.array([init_pa, init_pv, init_s, init_sv])
    return init_state


def dx_dt(state, t, params):
    # Parameters:
    f_hr_max = params["f_hr_max"]
    f_hr_min = params["f_hr_min"]
    r_tpr_max = params["r_tpr_max"]
    r_tpr_min = params["r_tpr_min"]
    ca = params["ca"]
    cv = params["cv"]
    k_width = params["k_width"]
    p_aset = params["p_aset"]
    tau = params["tau"]

    # Unknown parameters:
    i_ext = params["i_ext"]
    r_tpr_mod = params["r_tpr_mod"]
    sv_mod = params["sv_mod"]

    # State variables
    p_a = 100. * state[0]
    p_v = 10. * state[1]
    s = state[2]
    sv = 100. * state[3]

    # Building f_hr and r_tpr:
    f_hr = s * (f_hr_max - f_hr_min) + f_hr_min
    r_tpr = s * (r_tpr_max - r_tpr_min) + r_tpr_min - r_tpr_mod

    # Building dp_a/dt and dp_v/dt:
    dva_dt = -1. * (p_a - p_v) / r_tpr + sv * f_hr
    dvv_dt = -1. * dva_dt + i_ext
    dpa_dt = dva_dt / (ca * 100.)
    dpv_dt = dvv_dt / (cv * 10.)

    # Building dS/dt:
    ds_dt = (1. / tau) * (1. - 1. / (1 + np.exp(-1 * k_width * (p_a - p_aset))) - s)

    dsv_dt = i_ext * sv_mod

    # State derivative
    return np.array([dpa_dt, dpv_dt, ds_dt, dsv_dt])


def states_trajectory_to_sample(states, params):
    p_a = states[:, 0]
    p_v = states[:, 1]
    s = states[:, 2]

    f_hr_max = params["f_hr_max"]
    f_hr_min = params["f_hr_min"]
    f_hr = s * (f_hr_max - f_hr_min) + f_hr_min
    return np.stack((p_a, p_v, f_hr), axis=1)


def create_example(sample):
    length = sample.shape[0]
    fig, axs = plt.subplots(3)
    axs[0].plot(range(length), sample[:length, 0], 'r', label='Pa')
    axs[0].set(ylabel='Pa [mmHg]')
    axs[0].set(ylim=(0.0, 1.5))
    axs[1].plot(range(length), sample[:length, 1], 'b', label='Pv')
    axs[1].set(ylabel='Pv [mmHg]')
    axs[1].set(ylim=(0.0, 1.0))
    axs[2].plot(range(length), sample[:length, 2] * 60, 'k', label='f_hr')
    axs[2].set(ylabel='f_hr [bpm]')
    axs[2].set(ylim=(40, 200))

    for ax in axs:
        ax.set(xlabel='time')
        ax.grid()

    plt.savefig('CVS_example.png')


def get_random_params():
    i_ext = 0.0 if np.random.rand() > 0.5 else -2.0
    r_tpr_mod = 0.0 if np.random.rand() > 0.5 else 0.5

    return {"i_ext": i_ext,
            "r_tpr_mod": r_tpr_mod,
            "f_hr_max": 3.0,
            "f_hr_min": 2.0 / 3.0,
            "r_tpr_max": 2.134,
            "r_tpr_min": 0.5335,
            "sv_mod": 0.0001,
            "ca": 4.0,
            "cv": 111.0,

            # dS/dt parameters
            "k_width": 0.1838,
            "p_aset": 70,
            "tau": 20,
            "p_0lv": 2.03,
            "r_valve": 0.0025,
            "k_elv": 0.066,
            "v_ed0": 7.14,
            "T_sys": 4. / 15.,
            "cprsw_max": 103.8,
            "cprsw_min": 25.9
            }


def create_cvs_data(args):
    added_time = 50 if args.model == 'cvs' else 0
    t = np.arange(0.0, stop=(args.seq_len + added_time) * args.delta_t, step=args.delta_t)

    sample_size = 3
    state_size = 4

    raw_data = np.zeros(tuple([args.data_size, args.seq_len, sample_size]))
    latent_data = np.zeros((args.data_size, args.seq_len, state_size))
    params_data = []

    for i in range(args.data_size):
        # initial state
        init_state = init_random_state()
        params = get_random_params()
        params_data.append(params)

        states_trajectory = integrate.odeint(dx_dt, init_state, t, args=tuple([params]))[added_time:]

        raw_data[i] = states_trajectory_to_sample(states_trajectory, params)
        latent_data[i] = states_trajectory

    return raw_data, latent_data, params_data
