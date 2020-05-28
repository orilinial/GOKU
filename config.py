def load_data_config(args):
    if args.model == 'LV':
        args.output_dir = 'data/lv/'
        args.seq_len = 400
        args.data_size = 10000
        args.delta_t = 0.05
        args.noise_std = 0.01

    if args.model == 'CVS':
        args.output_dir = 'data/cvs/'
        args.seq_len = 400
        args.data_size = 1000
        args.delta_t = 1.0
        args.noise_std = 0.01

    if args.model == 'Pendulum':
        args.output_dir = 'data/pendulum/' if not args.friction else 'data/pendulum_friction/'
        args.seq_len = 200
        args.data_size = 500
        args.delta_t = 0.05
        args.noise_std = 0.0
        args.data_set = 1

    return args


def load_goku_train_config(args):
    if args.model == 'lv':
        args.num_epochs = 400
        args.mini_batch_size = 256
        args.seq_len = 100
        args.delta_t = 0.05
        args.data_path = 'data/lv/'
        args.norm = 'zscore'
        args.kl_annealing_epochs = 200
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001
        args.grounding_loss = 100.0

    if args.model == 'pixel_pendulum':
        args.num_epochs = 1600
        args.mini_batch_size = 128
        args.seq_len = 50
        args.delta_t = 0.05
        args.data_path = 'data/pendulum/'
        args.norm = 'zero_to_one'
        args.kl_annealing_epochs = 200
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001
        args.grounding_loss = 100.0

    if args.model == 'pixel_pendulum_friction':
        args.num_epochs = 1600
        args.mini_batch_size = 128
        args.seq_len = 50
        args.delta_t = 0.05
        args.data_path = 'data/pendulum_friction/'
        args.norm = 'zero_to_one'
        args.kl_annealing_epochs = 200
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001
        args.grounding_loss = 1000.0

    if args.model == 'cvs':
        args.num_epochs = 400
        args.mini_batch_size = 128
        args.seq_len = 200
        args.delta_t = 1.0
        args.data_path = 'data/cvs/'
        args.model = 'cvs'
        args.kl_annealing_epochs = 200
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001
        args.grounding_loss = 0.0

    return args


def load_latent_ode_train_config(args):
    if args.model == 'lv':
        args.num_epochs = 400
        args.mini_batch_size = 256
        args.seq_len = 100
        args.delta_t = 0.05
        args.data_path = 'data/lv/'
        args.norm = 'zscore'
        args.kl_annealing_epochs = 200
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001

    if args.model == 'pixel_pendulum':
        args.num_epochs = 1600
        args.mini_batch_size = 128
        args.seq_len = 50
        args.delta_t = 0.05
        args.data_path = 'data/pendulum/'
        args.norm = 'zero_to_one'
        args.kl_annealing_epochs = 200
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001

    if args.model == 'pixel_pendulum_friction':
        args.num_epochs = 1600
        args.mini_batch_size = 128
        args.seq_len = 50
        args.delta_t = 0.05
        args.data_path = 'data/pendulum_friction/'
        args.norm = 'zero_to_one'
        args.kl_annealing_epochs = 200
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001

    if args.model == 'cvs':
        args.num_epochs = 400
        args.mini_batch_size = 128
        args.seq_len = 200
        args.delta_t = 1.0
        args.data_path = 'data/cvs/'
        args.model = 'cvs'
        args.kl_annealing_epochs = 200
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001

    return args


def load_lstm_train_config(args):
    if args.model == 'lv':
        args.num_epochs = 400
        args.mini_batch_size = 256
        args.seq_len = 100
        args.data_path = 'data/lv/'
        args.norm = 'zscore'

    if args.model == 'pixel_pendulum':
        args.num_epochs = 1600
        args.mini_batch_size = 128
        args.seq_len = 50
        args.data_path = 'data/pendulum/'
        args.norm = 'zero_to_one'

    if args.model == 'pixel_pendulum_friction':
        args.num_epochs = 1600
        args.mini_batch_size = 128
        args.seq_len = 50
        args.data_path = 'data/pendulum_friction/'
        args.norm = 'zero_to_one'

    if args.model == 'cvs':
        args.num_epochs = 400
        args.mini_batch_size = 128
        args.seq_len = 200
        args.data_path = 'data/cvs/'
        args.model = 'cvs'

    return args
