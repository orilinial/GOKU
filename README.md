# GOKU - Deep Generative ODE Modelling with Known Unknowns

This repository is an implementation of the GOKU paper (link will be added soon).

### Data creation
To create the datasets used in the paper run:
* Friction-less pendulum:  `python3 create_data.py --model pendulum`
* Friction pendulum: `python3 create_data.py --model pendulum --friction`
* Double-pendulum experiment:  `python3 create_data.py --model double_pendulum`
* Cardiovascular system: `python3 create_data.py --model cvs`

The data would be created using default arguments. To view / modify them check the file `config.py`, and `create_data.py`.

### Training
To train the GOKU model run: `python3 goku_train.py --model <pendulum/pendulum_friction/double_pendulum/cvs>`

To train baselines:

* Latent-ODE: `python3 latent_ode_train.py --model <pendulum/pendulum_friction/double_pendulum/cvs>`.
* For Grounded Latent-ODE (L-ODE+), run Latent-ODE with the parameter `--grounding-loss`.
* LSTM: `python3 lstm_train.py --model <pendulum/pendulum_friction/double_pendulum/cvs>`.
* Direct-Identification (DI) has 3 different files for the different datasets (it cannot run the friction pendulum, since it needs the entire ODE functional form):
  * Pendulum: `python3 di_baseline_pendulum.py`
  * Double Pendulum: `python3 di_baseline_double_pendulum.py`
  * CVS: `python3 di_baseline_cvs.py`
  
### Requirements:
* python 3
* pytorch
* numpy
* gym (for the pendulum and double pendulum experiments)
