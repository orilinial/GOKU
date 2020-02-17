import numpy as np
from scipy import integrate
import argparse
import matplotlib.pyplot as plt
import torch
from utils.utils import set_seed, StatesToSamples
import os
import gym
from utils import gym_utils
from config import load_data_config


class OdeModel:
    def __init__(self, args, sample_size, state_size):
        self.sample_size = sample_size
        self.state_size = state_size
        self.seq_len = args.seq_len
        self.dtype = np.float64

    def init_random_state(self):
        raise NotImplementedError

    @staticmethod
    def dx_dt(state, t, params):
        raise NotImplementedError

    def states_trajectory_to_sample(self, states, params):
        raise NotImplementedError

    @staticmethod
    def create_example(sample):
        pass

    @staticmethod
    def get_random_params():
        raise NotImplementedError


class CVS(OdeModel):
    def __init__(self, args):
        super(CVS, self).__init__(args=args,
                                  sample_size=3,
                                  state_size=4)

    def init_random_state(self):
        max_sv = 1.0
        min_sv = 0.9

        max_pa = 85.0
        min_pa = 75.0

        max_pv = 7.0
        min_pv = 3.0

        max_s = 0.25
        min_s = 0.15

        init_sv = (np.random.rand() * (max_sv - min_sv) + min_sv)
        init_pa = (np.random.rand() * (max_pa - min_pa) + min_pa) / 100.0
        init_pv = (np.random.rand() * (max_pv - min_pv) + min_pv) / 10.0
        init_s = (np.random.rand() * (max_s - min_s) + min_s)

        init_state = np.array([init_pa, init_pv, init_s, init_sv])
        return init_state

    @staticmethod
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
        ds_dt = (1./tau) * (1. - 1./(1 + np.exp(-1 * k_width * (p_a - p_aset))) - s)

        dsv_dt = i_ext * sv_mod

        # State derivative
        return np.array([dpa_dt, dpv_dt, ds_dt, dsv_dt])

    def states_trajectory_to_sample(self, states, params):
        p_a = states[:, 0]
        p_v = states[:, 1]
        s = states[:, 2]

        f_hr_max = params["f_hr_max"]
        f_hr_min = params["f_hr_min"]
        f_hr = s * (f_hr_max - f_hr_min) + f_hr_min
        return np.stack((p_a, p_v, f_hr), axis=1)

    @staticmethod
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

    @staticmethod
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
                "T_sys": 4./15.,
                "cprsw_max": 103.8,
                "cprsw_min": 25.9
                }


class LV(OdeModel):
    def __init__(self, args):
        super(LV, self).__init__(args=args, sample_size=4, state_size=2)
        self.states_transform = StatesToSamples(sample_dim=self.sample_size, hidden_dim=10, state_dim=self.state_size)
        self.states_transform.load_state_dict(torch.load('utils/lv_emission_function.pkl'))

    def init_random_state(self):
        init_state = np.random.uniform(1.5, 3.0, size=self.state_size)
        return init_state

    @staticmethod
    def dx_dt(state, t, params):
        # Main parameters
        a = params['a']
        b = params['b']
        c = params['c']
        d = params['d']

        return np.array([a * state[0] - b * state[0] * state[1], -c * state[1] + d * state[0] * state[1]])

    def states_trajectory_to_sample(self, states, params):
        with torch.no_grad():
            sample = self.states_transform(torch.FloatTensor(states).unsqueeze(0)).squeeze().numpy()
        return sample

    @staticmethod
    def create_example(sample):
        length = 200

        plt.plot(range(length), sample[:length, 0], 'r', label='Prey')
        plt.plot(range(length), sample[:length, 1], 'b', label='Predator')

        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.ylabel('ODE state variables')
        plt.title('Example of LV ODE')
        plt.savefig('LV_example.png')

    @staticmethod
    def get_random_params():
        rand_params = np.random.uniform(1.0, 2.0, size=4)
        params = {"a": rand_params[0], "b": rand_params[1], "c": rand_params[2], "d": rand_params[3]}
        return params


def sample_gym(args, side=28):
    env_name = 'Pendulum-v0'
    env = gym.make(env_name).unwrapped
    env.seed(args.seed)
    data = np.zeros((args.data_size, args.seq_len, side, side))
    latent_data = np.zeros((args.data_size, args.seq_len, 2))
    params_data = []

    for trial in range(args.data_size):
        gym_utils.reset_env(env)
        params = gym_utils.get_params()  # This is a learned parameter - pendulum's length
        additional_params = gym_utils.get_unlearned_params()  # This parameter controls friction and isn't learned

        for step in range(args.seq_len):
            processed_frame = gym_utils.preproc(env.render('rgb_array'), side)
            data[trial, step] = processed_frame
            obs = gym_utils.step_env(args, env, [0.], params, additional_params)

            latent_data[trial, step, 0] = gym_utils.get_theta(obs)
            latent_data[trial, step, 1] = obs[-1]  # theta dot

        params_data.append(params)

    env.close()
    return data, latent_data, params_data


def create_raw_data(args):
    added_time = 50 if args.model == 'CVS' else 0
    t = np.arange(0.0, stop=(args.seq_len + added_time) * args.delta_t, step=args.delta_t)

    ode_model = eval(args.model)(args)
    if type(ode_model.sample_size) == int:
        raw_data = np.zeros(tuple([args.data_size, args.seq_len, ode_model.sample_size]), dtype=ode_model.dtype)
    elif type(ode_model.sample_size) == list:
        raw_data = np.zeros(tuple([args.data_size, args.seq_len] + ode_model.sample_size), dtype=ode_model.dtype)
    else:
        raise Exception('Please use a valid type of sample size (int or list of ints)')

    latent_data = np.zeros((args.data_size, args.seq_len, ode_model.state_size))
    params_data = []

    for i in range(args.data_size):
        # initial state
        init_state = ode_model.init_random_state()
        params = ode_model.get_random_params()
        params_data.append(params)

        states_trajectory = integrate.odeint(ode_model.dx_dt, init_state, t, args=tuple([params]))[added_time:]

        raw_data[i] = ode_model.states_trajectory_to_sample(states_trajectory, params)
        latent_data[i] = states_trajectory

    torch.save(params_data, args.output_dir + 'params_data.pkl')
    torch.save(raw_data, args.output_dir + 'raw_data.pkl')
    torch.save(latent_data, args.output_dir + 'latent_data.pkl')

    return raw_data, latent_data, params_data


def find_norm_params(data):
    mean = np.zeros(data.shape[2])
    std = np.zeros(data.shape[2])
    for feature in range(data.shape[2]):
        mean[feature] = data[:, :, feature].mean()
        std[feature] = data[:, :, feature].std()

    max_val = data.max()
    min_val = data.min()

    data_norm_params = {"mean": mean,
                        "std" : std,
                        "max" : max_val,
                        "min" : min_val}

    return data_norm_params


def add_noise(args, data):
    noisy_data = data + args.noise_std * np.random.normal(size=data.shape)
    return noisy_data


def create_mask(args, data_shape):
    revealed_n = int(round(args.mask_rate * args.seq_len))
    latent_mask = np.zeros(data_shape)
    max_val = int(args.seq_len * 0.75)
    for sample in range(data_shape[0]):
        train_latent_mask_ind = np.random.choice(range(max_val), size=revealed_n, replace=False)
        for mask_idx in train_latent_mask_ind:
            latent_mask[sample, mask_idx, :] = 1

    return latent_mask


def make_dataset(args):
    if args.new_dataset:
        if args.model == 'Pendulum':
            raw_data, latent_data, params_data = sample_gym(args)

        else:
            raw_data, latent_data, params_data = create_raw_data(args)

        buffer = int(round(raw_data.shape[0] * (1 - 0.1)))

        train_data = raw_data[:buffer]
        test_data = raw_data[buffer:]

        noisy_train_data = add_noise(args, train_data)
        noisy_test_data = add_noise(args, test_data)

        train_latent_data = latent_data[:buffer]
        test_latent_data = latent_data[buffer:]

        train_params_data = params_data[:buffer]
        train_params_data = {key: np.array([sample[key] for sample in train_params_data]) for key in train_params_data[0]}

        test_params_data = params_data[buffer:]
        test_params_data = {key: np.array([sample[key] for sample in test_params_data]) for key in test_params_data[0]}

        data_norm_params = find_norm_params(noisy_train_data)
        torch.save(train_params_data, args.output_dir + 'train_params_data.pkl')
        torch.save(test_params_data, args.output_dir + 'test_params_data.pkl')
        torch.save(train_latent_data, args.output_dir + 'train_latent_data.pkl')
        torch.save(test_latent_data, args.output_dir + 'test_latent_data.pkl')
        torch.save(data_norm_params, args.output_dir + 'data_norm_params.pkl')
        torch.save(test_data, args.output_dir + 'gt_test_data.pkl')

    else:
        train_latent_data = torch.load(args.output_dir + 'train_latent_data.pkl')
        test_latent_data = torch.load(args.output_dir + 'test_latent_data.pkl')

        dataset_dict = torch.load(args.output_dir + 'processed_data.pkl')
        noisy_train_data = dataset_dict['train']
        noisy_test_data = dataset_dict['test']

    train_latent_mask = create_mask(args, train_latent_data.shape)
    test_latent_mask = create_mask(args, test_latent_data.shape)

    dataset_dict = {'train': noisy_train_data,
                    'train_latent': train_latent_data * train_latent_mask,
                    'train_latent_mask': train_latent_mask,
                    'test': noisy_test_data,
                    'test_latent': test_latent_data * test_latent_mask,
                    'test_latent_mask': test_latent_mask}

    torch.save(dataset_dict, args.output_dir + 'processed_data.pkl')

    args_dict = {'mask_rate': args.mask_rate, 'noise_std': args.noise_std, 'model': args.model}
    torch.save(args_dict, args.output_dir + 'data_args.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--seq-len', type=int)
    parser.add_argument('--data-size', type=int)
    parser.add_argument('--delta-t', '-dt', type=float)
    parser.add_argument('--noise-std', type=float)
    parser.add_argument('--mask-rate', type=float, default=0.01)
    parser.add_argument('--model', choices=['CVS', 'LV', 'Pendulum'], required=True)
    parser.add_argument('--new-dataset', action='store_true')
    parser.add_argument('--friction', action='store_true')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    args = load_data_config(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args.seed)
    make_dataset(args)
