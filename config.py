def load_data_config(args):
    if args.model == 'cvs':
        args.output_dir = 'data/cvs/'
        args.seq_len = 400
        args.data_size = 1000
        args.delta_t = 1.0
        args.noise_std = 0.05
        args.seed = 12

    if args.model == 'pendulum':
        args.output_dir = 'data/pendulum/' if not args.friction else 'data/pendulum_friction/'
        args.seq_len = 100
        args.data_size = 500
        args.delta_t = 0.05
        args.noise_std = 0.0
        args.seed = 13

    if args.model == 'double_pendulum':
        args.output_dir = 'data/double_pendulum/'
        args.seq_len = 100
        args.data_size = 500
        args.delta_t = 0.05
        args.noise_std = 0.0
        args.seed = 13

    return args


def load_goku_train_config(args):
    if args.model == 'pendulum':
        args.num_epochs = 1600
        args.mini_batch_size = 64
        args.seq_len = 50
        args.delta_t = 0.05
        args.data_path = 'data/pendulum/'
        args.norm = 'zero_to_one'
        args.kl_annealing_epochs = 200
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001
        args.grounding_loss = 100.0

    if args.model == 'double_pendulum':
        args.num_epochs = 1600
        args.mini_batch_size = 64
        args.seq_len = 50
        args.delta_t = 0.05
        args.data_path = 'data/double_pendulum/'
        args.norm = 'zero_to_one'
        args.kl_annealing_epochs = 200
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001
        args.grounding_loss = 100.0

    if args.model == 'pendulum_friction':
        args.num_epochs = 1600
        args.mini_batch_size = 64
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

    args.seed = 14
    return args


def load_latent_ode_train_config(args):
    if args.model == 'pendulum':
        args.num_epochs = 1600
        args.mini_batch_size = 64
        args.seq_len = 50
        args.delta_t = 0.05
        args.data_path = 'data/pendulum/'
        args.norm = 'zero_to_one'
        args.kl_annealing_epochs = 200
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001

    if args.model == 'double_pendulum':
        args.num_epochs = 1600
        args.mini_batch_size = 64
        args.seq_len = 50
        args.delta_t = 0.05
        args.data_path = 'data/double_pendulum/'
        args.norm = 'zero_to_one'
        args.kl_annealing_epochs = 200
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001

    if args.model == 'pendulum_friction':
        args.num_epochs = 1600
        args.mini_batch_size = 64
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

    args.seed = 14
    return args


def load_lstm_train_config(args):
    if args.model == 'pendulum':
        args.num_epochs = 1000
        args.mini_batch_size = 64
        args.seq_len = 50
        args.data_path = 'data/pendulum/'
        args.norm = 'zero_to_one'

    if args.model == 'double_pendulum':
        args.num_epochs = 1000
        args.mini_batch_size = 64
        args.seq_len = 50
        args.data_path = 'data/double_pendulum/'
        args.norm = 'zero_to_one'

    if args.model == 'pendulum_friction':
        args.num_epochs = 1000
        args.mini_batch_size = 64
        args.seq_len = 50
        args.data_path = 'data/pendulum_friction/'
        args.norm = 'zero_to_one'

    if args.model == 'cvs':
        args.num_epochs = 400
        args.mini_batch_size = 128
        args.seq_len = 200
        args.data_path = 'data/cvs/'
        args.model = 'cvs'

    args.seed = 14
    return args
